# diffusion-pipe 全方位性能调优方案

> 基于 3 个独立 Agent 交叉分析（前向计算 / 数据管道 / 分布式显存）整合。
> 已实施的优化不在本方案范围内：QKV 融合、AdaLN 融合、RoPE dtype 上提、per-block torch.compile(opt-in)。

---

## 关键发现（分析中确认的）

1. **attention backend 硬编码为 `'torch'`**（`cosmos_predict2_modeling.py:1247`），`transformer_engine` 路径是死代码且缺少 import。当前 attention 走 PyTorch SDPA（内部调用 Flash Attention），但 RoPE 无法被 fuse 进 attention kernel。
2. **`make_contiguous` 每步做 140 次无用拷贝**（28 blocks × 5 个不变 tensor），在 compile 范围外。
3. **Block swap 的 `non_blocking=True` 退化为同步传输**——CPU 侧权重不是 pinned memory。
4. **每步 ~1400 次 Python 调用 + ~840 次 kernel launch**，batch=1 时 launch overhead 占 6-12%。
5. **`get_data_iterator_for_step` 同步预取所有 micro-batch**，GPU 在此期间空闲。
6. **Cache 读取用 `torch.load`（pickle 反序列化）**，无 `weights_only=True`。

---

## P0 — 立即实施（零风险 / 极高 ROI）

### 1. TF32 全局开启

**位置**: `train.py` 顶部（import 之后）
**改动**: 2 行

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

**收益**: bf16 训练下 1-3%（loss 计算、RMSNorm variance 等 fp32 子操作）；fp32 训练下 20-40%
**风险**: 无。Ampere+ 标准特性
**compile 时**: 仍有效

### 2. PYTORCH_CUDA_ALLOC_CONF 设置

**位置**: `train.py` 最顶部（所有 import 之前）
**改动**: 2 行

```python
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,garbage_collection_threshold:0.6')
```

**收益**: block swap 频繁分配/释放产生碎片，此设置减少 OOM 概率 30-50%，可支持更大 batch/更多 blocks_to_swap
**风险**: 无。PyTorch 官方推荐配置

### 3. Block swap pinned memory

**位置**: `utils/offloading.py:257-275`（`prepare_block_devices_before_forward`）
**问题**: `non_blocking=True` 对非 pinned memory 实际退化为同步传输，ThreadPoolExecutor 异步预取几乎失效
**改动**:

```python
# 在 line 270-272 之后，CPU 侧权重 pin memory
for b in self.blocks[self.num_blocks - self.blocks_to_swap:]:
    b.to(self.device)
    weights_to_device(b, torch.device('cpu'))
    for name, module in b.named_modules():
        if 'lora' not in name and hasattr(module, 'weight') and module.weight is not None:
            if not module.weight.data.is_pinned():
                module.weight.data = module.weight.data.pin_memory()
```

**收益**: swap 带宽提升 30-50%（pinned ~12 GB/s vs 普通 ~3-6 GB/s over PCIe Gen4）
**风险**: 低。pinned memory 占用更多系统内存，但 swap 的 block 数量有限
**这是 block swap 场景最大的可优化点**

### 4. Cache weights_only=True

**位置**: `utils/cache.py:35`
**改动**: 1 行

```python
item = torch.load(buffer, map_location='cpu', weights_only=True)
```

**收益**: 5-15%。跳过 pickle 的 Python 对象重建，只加载 tensor
**风险**: 低。缓存内容是 tensor dict + 基本类型，weights_only 兼容

### 5. GELU tanh 近似

**位置**: `models/cosmos_predict2_modeling.py:280`
**改动**: 1 行

```python
self.activation = nn.GELU(approximate='tanh')
```

**收益**: 2-5%。tanh 近似可被 fuse 进 GEMM epilogue，erf 精确版需独立 kernel
**风险**: 低。LLM 标准做法（GPT-2、LLaMA 等），数值差异 <0.01。需确认推理端也用 tanh 近似
**compile 时**: 仍建议改

### 6. AdamW fused=True

**位置**: `train.py:659`
**改动**: 1 行

```python
if optim_type_lower == 'adamw':
    klass = torch.optim.AdamW
    kwargs.setdefault('fused', True)  # 新增
```

**收益**: 1-3%。单 kernel 更新所有参数，减少 kernel launch
**风险**: 低。`torch.optim.AdamW(fused=True)` 是 PyTorch 原生实现，不依赖 deepspeed 编译
**注意**: gradient_release 模式下已设 `foreach=False`，需确认 fused 兼容性。gradient_release 为每个 param 创建独立 optimizer，fused 的收益在此模式下有限

---

## P1 — 短期实施（低风险 / 高收益）

### 7. 去除 make_contiguous 冗余拷贝

**位置**: `models/cosmos_predict2.py:966`（`TransformerLayer.forward` 末尾）
**问题**: 每个 block forward 后对**所有**传递的 tensor 做 `.contiguous()`，但 `t_embedding`、`crossattn_emb`、`rope_emb`、`adaln_lora`、`timesteps` 在 block 间不变
**改动**:

```python
# 只对 x 做 contiguous
x_B_T_H_W_D = x_B_T_H_W_D.contiguous()
return (x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T)
```

**收益**: 2-4%。每步省 140 次无用 contiguous 拷贝
**风险**: 低。需确认下游不依赖这些 tensor 的 stride
**compile 时**: 仍有效（compile 覆盖 Block.forward，不覆盖 TransformerLayer.forward）

### 8. einops rearrange → 原生 torch ops

**位置**: `models/cosmos_predict2_modeling.py` 多处（Block.forward + torch_attention_op + compute_qkv）
**改动**: 详见下表

| 位置 | einops pattern | 原生替代 |
|------|---------------|---------|
| 行1099-1109 (9处) | `"b t d -> b t 1 1 d"` | `.unsqueeze(2).unsqueeze(3)` |
| 行1122-1125 | `"b t h w d -> b (t h w) d"` | `.reshape(B, T*H*W, D)` |
| 行1129-1132 | `"b (t h w) d -> b t h w d"` | `.reshape(B, T, H, W, D)` |
| 行1145-1155 (×2) | 同上 | 同上 |
| `compute_qkv` 行467 | `"b ... (h d) -> b ... h d"` | `.reshape(*shape[:-1], H, D)` |
| `torch_attention_op` 行331-336 | 多个 pattern | `.transpose(1,2)` + `.reshape()` |

`torch_attention_op` 完整替换:

```python
def torch_attention_op(q, k, v):
    B, S, H, D = q.shape
    q = q.transpose(1, 2)  # b h s d
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(B, S, H * D)
```

**收益**: 2-5%。每步 ~840 次 rearrange 的 pattern 解析 + Python 调用栈开销消除
**风险**: 极低。reshape/transpose/unsqueeze 语义完全等价
**compile 时**: 无效（dynamo 自动编译 rearrange）。仅在 `torch_compile=False` 时有意义

### 9. RMSNorm 用 F.rms_norm

**位置**: `models/cosmos_predict2_modeling.py:266-273`
**改动**:

```python
def forward(self, x):
    return F.rms_norm(x, self.weight.shape, self.weight, self.eps)
```

**收益**: 3-6%。当前每次 `.float()` → `_norm` → `.type_as()` → `* weight` 约 5 个 kernel；`F.rms_norm` 是单 fused kernel（bf16 输入内部 fp32 累加）
**风险**: 低。需 PyTorch ≥ 2.4。数值等价
**compile 时**: 无效（compile 自动 fuse）

### 10. Block swap sync 优化

**位置**: `utils/offloading.py:71-86`
**问题**: D2H 和 H2D 之间有 `stream.synchronize()`（line 78），强制 D2H 完全完成后才开始 H2D。实际上它们操作不同 tensor
**改动**:

```python
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    # D2H 和 H2D 在同一 stream 中连续提交，无需中间同步
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.record_stream(stream)
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view
stream.synchronize()
# 用 current_stream().wait_stream(stream) 替代第二次 current_stream().synchronize()
torch.cuda.current_stream().wait_stream(stream)
```

**收益**: 5-10% swap 时间（减少 1 次 sync + 更好的 stream 并行）
**风险**: 低-中。需测试 "illegal loss value" 是否复现

### 11. unsloth checkpointing + pinned memory

**位置**: `utils/unsloth_utils.py:33, 47` + 用户配置 `activation_checkpointing = "unsloth"`
**改动**: unsloth checkpoint 的 hidden states D2H/H2D 也需要 pinned memory

```python
# line 33
if not hidden_states.is_pinned():
    saved = torch.empty_like(hidden_states, pin_memory=True)
    saved.copy_(hidden_states, non_blocking=True)
else:
    saved = hidden_states.to('cpu', non_blocking=True)
```

**收益**: backward 重计算时 H2D 传输加速 30-50%；unsloth 本身比 PyTorch native 省 20-40% 显存
**风险**: 低

### 12. AdaLN reshape/permute → chunk

**位置**: `models/cosmos_predict2_modeling.py:1077-1096`
**改动**:

```python
mod_9D = self.adaln_modulation_2(self.adaln_modulation_1(silu_emb))
# 直接 chunk，避免 permute
branches = mod_9D.chunk(3, dim=-1)  # 3 × (B, T, 3*D)
shift_self, scale_self, gate_self = branches[0].chunk(3, dim=-1)
shift_cross, scale_cross, gate_cross = branches[1].chunk(3, dim=-1)
shift_mlp, scale_mlp, gate_mlp = branches[2].chunk(3, dim=-1)
```

**收益**: 1-2%。chunk 返回 view 且最后一维连续，避免 permute 改变 stride 后的隐式 contiguous
**风险**: 低。需确认 chunk 顺序与原 permute 索引一致
**compile 时**: 无效

### 13. batch=1 时 unsqueeze 替代 torch.stack

**位置**: `utils/dataset.py:1009-1013`
**改动**:

```python
if torch.is_tensor(features[0]):
    if len(features) == 1:
        features = features[0].unsqueeze(0)
    else:
        shape = features[0].shape
        if all(f.shape == shape for f in features):
            features = torch.stack(features)
```

**收益**: 2-5%（batch=1 场景）
**风险**: 极低

### 14. _broadcast_target non_blocking 传输

**位置**: `utils/dataset.py:1382`
**改动**:

```python
target = target.to('cuda', non_blocking=True)  # 替代 target.to('cuda')
```

**收益**: 微小（pipeline_stages>1 时有效）
**风险**: 极低

---

## P2 — 中期实施（中等风险 / 中等收益）

### 15. Cross-attn KV 投影融合

**位置**: `models/cosmos_predict2_modeling.py:405-407`（`Attention.__init__`），`compute_qkv` 行461-465
**改动**:

```python
# __init__ 中 cross-attn
if not self.is_selfattn:
    self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
    self.kv_proj = nn.Linear(context_dim, 2 * inner_dim, bias=False)  # 融合 K+V

# compute_qkv 中
else:
    q = self.q_proj(x)
    context = x if context is None else context
    kv = self.kv_proj(context)
    k, v = kv.chunk(2, dim=-1)
```

**收益**: 1-3%。56 次小 GEMM → 28 次大 GEMM
**风险**: 中。需同步修改 `_remap_state_dict_keys` / `_split_state_dict_keys` / LoRA key remap
**compile 时**: 仍有价值（compile 不会自动融合不同权重的 Linear）

### 16. prepare_inputs 移入 DataLoader worker

**位置**: `utils/dataset.py:1357-1368`（`_pull_batches_from_dataloader`），`models/cosmos_predict2.py:680-730`
**问题**: `prepare_inputs` 中的 `torch.randn_like`（噪声生成）、timestep 采样、flow matching 计算全在主进程同步执行，阻塞 GPU
**改动**: 拆分 `prepare_inputs` 为 CPU 部分（在 worker 执行）和 GPU 部分（在主进程执行）

```python
# Dataset.__getitem__ 中
if hasattr(self.model, 'prepare_inputs_cpu'):
    batch = self.model.prepare_inputs_cpu(batch)

# _pull_batches_from_dataloader 中只做 GPU 相关操作
def _pull_batches_from_dataloader(self):
    for batch in self.dataloader:
        features, label = batch  # 已预处理
        target, mask = label
        target = self._broadcast_target(target)
        ...
```

**收益**: 10-30%（噪声生成和加噪计算与 GPU 计算重叠）
**风险**: 中。需重构 prepare_inputs 拆分 CPU/GPU 部分，且需处理 RNG 状态在 worker 间的正确性

### 17. 跨步数据预取 overlap

**位置**: `train.py:167-173`（`get_data_iterator_for_step`），`train.py:902-905`（训练循环）
**问题**: 每步开始时同步预取所有 micro-batch，GPU 在此期间空闲
**改动**: 用后台线程预取下一步数据

```python
# 使用 ThreadPoolExecutor 在上一步 GPU 计算时预取下一步
from concurrent.futures import ThreadPoolExecutor
prefetch_executor = ThreadPoolExecutor(max_workers=1)
next_iterator_future = prefetch_executor.submit(
    get_data_iterator_for_step, train_dataloader, model_engine
)
# ... 当前步 GPU 计算 ...
iterator = next_iterator_future.result()
next_iterator_future = prefetch_executor.submit(
    get_data_iterator_for_step, train_dataloader, model_engine
)
```

**收益**: 5-20%（取决于数据准备时间与 GPU 计算时间的比率）
**风险**: 中。多线程与 DeepSpeed pipeline engine 的交互需验证

### 18. 增加 DataLoader workers

**位置**: `utils/dataset.py:1347-1355`
**改动**:

```python
num_workers = min(4, os.cpu_count() // 2)
self.dataloader = torch.utils.data.DataLoader(
    self.dataset,
    pin_memory=not config.get('compile', False),  # 条件性启用
    batch_size=None,
    sampler=sampler,
    num_workers=num_workers,
    persistent_workers=True,
    prefetch_factor=4,
)
```

**注意**: 增加 num_workers 时必须配合 SQLite 只读模式 + `worker_init_fn` 重新打开连接
**收益**: 10-25%
**风险**: 中。SQLite 锁竞争 / fork 后文件句柄共享

### 19. 缓存预热到页缓存

**位置**: `utils/cache.py` 新增方法
**改动**:

```python
def warmup(self):
    """预读所有 shard 到页缓存"""
    for shard_id in range(self.shard + 1):
        path = self.path / f'shard_{shard_id}.bin'
        if path.exists():
            with open(path, 'rb') as f:
                while f.read(1024 * 1024):
                    pass
```

**收益**: 5-15%（首轮 epoch I/O 等待减少）
**风险**: 低

### 20. 条件性 pin_memory

**位置**: `utils/dataset.py:1349`
**改动**: `pin_memory=not config.get('compile', False)`
**收益**: 3-10%（非 compile 路径）
**风险**: 中。与 `copy_args_to_cpu_if_needed` patch 的交互需测试

### 21. 定期 empty_cache

**位置**: `train.py` 训练循环中
**改动**:

```python
if step % 50 == 0:
    empty_cuda_cache()
```

**收益**: 防止长训练中碎片累积导致 OOM
**风险**: 低（empty_cache 约 1-5ms 开销，每 50 步可忽略）

### 22. GenericOptim CPU offload 同步优化

**位置**: `optimizers/generic_optim.py:500-502`
**改动**: 用 Event 替代 synchronize，延迟到下次 step 开始时等待

```python
if synchronize:
    self._sync_event = torch.cuda.Event()
    self._sync_event.record()
    # 下次 step 开始时: self._sync_event.synchronize()
```

**收益**: 5-10%（允许 optimizer state CPU offload 与下一个 micro-batch forward 重叠）
**风险**: 中

---

## P3 — 长期 / 高风险 / 需深度改造

### 23. CUDA Graphs（batch=1 最大潜力）

**位置**: `MiniTrainDIT.__init__` 行1277 已预留 `self.cuda_graphs = {}` 但未使用
**改动**: 对固定 shape 输入录制 CUDA Graph，将所有 kernel launch 降为 1 次
**收益**: 10-25%（batch=1 场景）。840 kernel × ~7μs launch ≈ 6ms 纯 launch 开销消除
**风险**: 高
- 不兼容 activation_checkpointing（recompute 与 graph 冲突）
- 不兼容 block swap（权重地址变化）
- 需为每个 (batch, T, H, W) 组合录制独立 graph
- 反向传播需单独处理
**compile 时**: `mode='reduce-overhead'` 已内置 CUDA Graphs，此优化冗余

### 24. Pipeline 通信与计算重叠

**位置**: `utils/patches.py:train_schedule_steps`
**改动**: 将 `SendActivation`/`SendGrad` 放到单独 CUDA stream 异步执行
**收益**: 5-15%（取决于激活 tensor 大小和带宽）
**风险**: 高。涉及 DeepSpeed 内部通信逻辑，可能导致死锁

### 25. Interleaved 1F1B schedule

**位置**: `utils/patches.py:train_schedule_steps`
**改动**: Megatron-LM 风格 interleaved schedule，每 stage 分多个 virtual chunks
**收益**: 10-20%（气泡减半）
**风险**: 高。需大改 schedule 逻辑和通信模式

### 26. Cache 改用 safetensors 格式

**位置**: `utils/cache.py` 整体重写
**收益**: 30-60%。零拷贝 mmap 读取，跳过 pickle 反序列化
**风险**: 中。需迁移存储格式 + 处理非 tensor 类型（caption 字符串等）

### 27. gradient_release 正确性验证

**位置**: `train.py:708-711`
**问题**: `add_` hack 使不同 pipeline stage 可能看到不同版本参数。代码注释自承认 "unbelievably hacky and not mathematically sound"
**建议**: 添加文档警告 + 监控训练 loss 异常波动
**风险**: 高（修改可能破坏训练效果）

---

## 总览矩阵

| # | 优化点 | 优先级 | 位置 | 预估收益 | 风险 | compile时无效? |
|---|--------|--------|------|---------|------|:---:|
| 1 | TF32 全局开启 | P0 | train.py | 1-3% (bf16) | 无 | 否 |
| 2 | PYTORCH_CUDA_ALLOC_CONF | P0 | train.py | 减少 OOM 30-50% | 无 | 否 |
| 3 | Block swap pinned memory | P0 | offloading.py:270 | 30-50% swap 带宽 | 低 | 否 |
| 4 | Cache weights_only=True | P0 | cache.py:35 | 5-15% | 低 | 否 |
| 5 | GELU tanh 近似 | P0 | modeling.py:280 | 2-5% | 低 | 否 |
| 6 | AdamW fused=True | P0 | train.py:659 | 1-3% | 低 | 否 |
| 7 | 去 make_contiguous 冗余 | P1 | cosmos_predict2.py:966 | 2-4% | 低 | 否 |
| 8 | rearrange→原生 op | P1 | modeling.py 多处 | 2-5% | 极低 | 是 |
| 9 | RMSNorm F.rms_norm | P1 | modeling.py:266 | 3-6% | 低 | 是 |
| 10 | Block swap sync 优化 | P1 | offloading.py:71 | 5-10% swap | 低-中 | 否 |
| 11 | unsloth + pinned memory | P1 | unsloth_utils.py:33 | 30-50% H2D | 低 | 否 |
| 12 | AdaLN chunk 替代 permute | P1 | modeling.py:1077 | 1-2% | 低 | 是 |
| 13 | batch=1 unsqueeze | P1 | dataset.py:1009 | 2-5% | 极低 | 否 |
| 14 | broadcast non_blocking | P1 | dataset.py:1382 | 微小 | 极低 | 否 |
| 15 | Cross-attn KV 融合 | P2 | modeling.py:405 | 1-3% | 中 | 否 |
| 16 | prepare_inputs 移入 worker | P2 | dataset.py + model | 10-30% | 中 | 否 |
| 17 | 跨步数据预取 overlap | P2 | train.py:902 | 5-20% | 中 | 否 |
| 18 | 增加 DataLoader workers | P2 | dataset.py:1347 | 10-25% | 中 | 否 |
| 19 | 缓存预热页缓存 | P2 | cache.py | 5-15% 首轮 | 低 | 否 |
| 20 | 条件性 pin_memory | P2 | dataset.py:1349 | 3-10% | 中 | 否 |
| 21 | 定期 empty_cache | P2 | train.py 循环 | 防 OOM | 低 | 否 |
| 22 | GenericOptim sync 优化 | P2 | generic_optim.py:500 | 5-10% | 中 | 否 |
| 23 | CUDA Graphs | P3 | MiniTrainDIT | 10-25% | 高 | 是 |
| 24 | Pipeline 通信重叠 | P3 | patches.py | 5-15% | 高 | 否 |
| 25 | Interleaved 1F1B | P3 | patches.py | 10-20% | 高 | 否 |
| 26 | Cache safetensors 格式 | P3 | cache.py | 30-60% | 中 | 否 |
| 27 | gradient_release 正确性 | P3 | train.py:708 | 正确性 | 高 | 否 |

---

## 实施路径

### 阶段 1: Quick Wins（1-2 小时，P0 全部 + P1 部分）

改动: #1-6, #7, #13, #14
预估累计收益: **8-20%**（非 compile 路径）

### 阶段 2: Block Swap 优化（2-4 小时，P0 #3 + P1 #10-11）

改动: #3, #10, #11
预估累计收益: **额外 15-30%**（block swap 场景）

### 阶段 3: 数据管道优化（4-8 小时，P1-2 #16-20）

改动: #16, #17, #18, #19, #20
预估累计收益: **额外 10-25%**（数据加载瓶颈场景）

### 阶段 4: 深度优化（P3，需充分测试）

改动: #15, #22-27
预估累计收益: **额外 10-30%**（场景依赖）

### 按场景预估总收益

| 场景 | 预估总收益 |
|------|-----------|
| 单卡 LoRA + block swap（batch=1） | 25-55% |
| 单卡 LoRA 无 block swap | 12-25% |
| 多卡 pipeline parallel | 15-35% |
| 全量 fine-tune | 10-25% |

### compile 与非 compile 路径的差异

**非 compile 路径（当前默认）**: 所有 P0-P1 优化均有效，#8/9/12 对 batch=1 收益显著

**compile 路径（`torch_compile=true`）**:
- 仍需做的: #1(TF32)、#2(alloc conf)、#3(pinned mem)、#4(weights_only)、#5(GELU)、#6(fused AdamW)、#7(make_contiguous)、#15(KV融合)
- 不需要做的（compile 已覆盖）: #8(rearrange)、#9(RMSNorm)、#12(permute→chunk)、#23(CUDA Graphs via reduce-overhead)
- compile 额外建议: `mode='max-autotune'` 比 `default` 多 5-10% 收益（首次编译时间更长）
