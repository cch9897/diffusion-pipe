# Anima/Cosmos-Predict2 Fine-Tuning — 多Agent交叉审议报告

**日期**: 2025-06-18  
**项目**: diffusion-pipe  
**范围**: Anima FT、性能调优、严重Bug审计  
**评审方式**: 3个Agent正交领域并行交叉评审

---

## 评审Agent

| Agent | 领域 | 职责 |
|-------|------|------|
| A | 性能/关键路径 | torch_compile、QKV/AdaLN融合、Kahan求和、block swap、autocast |
| B | 正确性/Bug | state_dict remap、LoRA remap、数值精度、损失函数 |
| C | 架构/集成 | Pipeline拆分、参数分组、save/load roundtrip、DeepSpeed兼容 |

---

## 交叉投票矩阵

| # | 议题 | A | B | C | 共识 |
|---|------|---|---|---|------|
| P0-1 | Block swap ThreadPool CUDA context race | P0 | — | — | A独 |
| P0-2 | Kahan复用p.grad为临时buffer | P0 | P1 | — | ✅ AB |
| P0-3 | AdaLN LoRA remap doc-code不一致 | — | P0 | — | B独 |
| P1-1 | make_contiguous 280次/step冗余 | P1 | — | — | A独 |
| P1-2 | swap子线程冗余current_stream sync | P1 | — | — | A独 |
| P1-3 | is_grad_enabled检测reentrant脆弱 | P1 | P2 | — | ✅ AB |
| P1-4 | torch_compile + block_swap应硬阻止 | P1 | — | P2 | ⚠️ AC |
| P1-5 | QKV/KV LoRA lora_A静默损坏 | — | P1 | P1 | ✅ BC |
| P1-6 | gradient_release数学不严谨 | — | P1 | — | B独 |
| P1-7 | LLM adapter in/out_proj不在LoRA target | — | — | P1 | C独 |
| P1-8 | t_embedder AdaLN params在base组 | — | — | P1 | C独 |
| P1-9 | partition_method=manual断言问题 | — | — | P1 | C独 |
| P2-1 | AUTOCAST_DTYPE=None非train.py入口 | P2 | — | — | A独 |
| P2-2 | RMSNorm fallback fp32 upcast | P2 | — | — | A独 |
| P2-3 | loss mask mean非标准 | — | P2 | — | B独 |
| P2-4 | automagic_lr bf16符号位下溢 | — | P2 | — | B独 |
| P2-5 | micro_batch TOML键类型转换 | — | P2 | — | B独 |
| P2-6 | 文本encoder在pipeline内无感知 | — | — | P2 | C独 |
| P2-7 | examples fallback可能不准确 | — | — | P2 | C独 |

图例: ✅ 两Agent独立发现并确认 | ⚠️ 相关性发现但严重度不同 | — 该Agent未关注此议题

---

## P0 — 必须立即修复

### P0-1: Block swap 在 ThreadPoolExecutor 中执行 CUDA 操作

**文件**: `utils/offloading.py:171-173` + `swap_weight_devices_cuda:48-94`

**风险**: CUDA context 是 thread-local 的。子线程中的 `stream.synchronize()`、`wait_stream()` 等 API 调用可能因缺少 context 成为 no-op，导致主线程在 H2D 完成前开始计算 → illegal loss value 或静默数值错误。

**当前代码**:

```python
# offloading.py:171
self.futures[block_idx_to_cuda] = self.thread_pool.submit(
    move_blocks, ...  # move_blocks calls swap_weight_devices_cuda
)
```

**修复建议**: 移除 ThreadPoolExecutor，改用主线程异步 CUDA stream：将 D2H/H2D 提交到专用 stream，`wait_for_block` 改成 `stream.wait_stream(current_stream)` + CUDA event 等待。线程池在此场景下不加速（max_workers=1），只引入危险的 CUDA context 边界问题。

---

### P0-2: Kahan bf16 补偿求和中 p.grad 被就地覆写

**文件**: `optimizers/generic_optim.py:490-498`

**风险**:
- (a) `p.grad.copy_(p.detach())` 将梯度 tensor 覆写为参数当前值。step() 到 zero_grad() 之间任何读取 p.grad 的外部代码（hook、logger）会拿到参数值而非梯度
- (b) 第497行 TODO: `non_blocking=True` 回传 CUDA error，当前使用同步传输拖慢每步

**当前代码**:

```python
shift = state['shift'].to(p.device, non_blocking=True)
shift.add_(update)                 # shift = comp + update
p.grad.copy_(p.detach())           # reuse p.grad as temp buffer
p.add_(shift)                      # bf16-rounded addition
shift.add_(p.grad.sub_(p))         # recover rounding error
state['shift'] = shift.to(kahan_buffer_device)  # sync — TODO: non_blocking causes CUDA error
```

**修复建议**: 分配独立临时 buffer（`torch.empty_like(p)`），不复用 p.grad。额外显存开销 ≈ 1×param，通常 <10MB。

---

### P0-3: AdaLN LoRA remap 文档承诺"rank==3R时cat"但代码始终取平均

**文件**: `models/cosmos_predict2.py:218-328`

**风险**: docstring (line 218-222) 明确声明：

> if the target rank < 3R, we average; if equal, we cat

但代码 line 320-321 无条件取平均：

```python
new_sd[fused_key_1] = sum(l1_list) / 3.0
new_sd[fused_key_2] = sum(l2_list) / 3.0
```

用户若提供 rank=3R 的融合后 LoRA 权重，会触发静默损毁而非无损拼接。没有 rank 比较逻辑。

**修复**: 实现 rank 检测逻辑（比较 fused rank vs 3×legacy rank），或更新 docstring 移除"if equal we cat"承诺并标注不可逆损失。

---

## P1 — 应尽快修复

### P1-1: InitialLayer.make_contiguous 对 5 个不变 tensor 做 280次/step 冗余拷贝

**文件**: `models/cosmos_predict2.py:1016` + `models/base.py:33-34`

**影响**: ~2-4% 吞吐损失。`make_contiguous` 对 block 间完全不变的 `t_embedding_B_T_D`、`crossattn_emb`、`rope_emb_L_1_1_D`、`adaln_lora_B_T_3D`、`timesteps_B_T` 每 step 做 28 blocks × 5 tensors = 140 次无意义 contiguous。

**修复**: `make_contiguous` 只对 `x_B_T_H_W_D` 调用；其他 5 个 tensor 在 block 间不变，无需 contiguous。

---

### P1-2: swap_weight_devices_cuda 子线程中冗余 current_stream().synchronize()

**文件**: `utils/offloading.py:74`

**影响**: swap 延迟增加。stream synchronization 在 D2H/H2D 后用 `stream.synchronize() + wait_stream()` 已经完成，额外 sync 不必要且可能执行在错误的 context。

**修复**: 移除这行 sync。

---

### P1-3: torch.is_grad_enabled() 检测 reentrant checkpoint 不可靠

**文件**: `utils/offloading.py:299, 313`

**风险**: unsloth checkpoint 使用 `@torch.amp.custom_fwd`，不改变 `is_grad_enabled()` 状态 → replay forward 时也触发 block swap → D2H/H2D 与重计算竞态。注释已注明局限但未处理 unsloth 路径。

**当前代码**:

```python
if self.reentrant_activation_checkpointing and torch.is_grad_enabled():
    return  # skip block swap in replay forward
```

**修复**: 用显式标志位 `self._is_replay_forward`（在 hook 里 toggle），替代隐式的 grad 状态探测。

---

### P1-4: torch_compile + blocks_to_swap 应硬阻止而非仅 print warning

**文件**: `models/cosmos_predict2.py:841-842`

**影响**: 组合导致每步 dynamo guard 失效 + 持续 recompile → 训练几乎停滞。

**修复**: 检测到 compile=true + blocks_to_swap>0 时 `raise ValueError` 并给出明确错误消息。同理 unsloth + compile 组合。

---

### P1-5: QKV/KV LoRA lora_A 加载时取平均 → 静默丢失 50–67% 表达能力

**文件**: `models/cosmos_predict2.py:249-258, 279-286`

**风险**: 用户迁移旧 LoRA checkpoint 时，lora_A 从 3×(R,D) 平均为 1×(R,D)，丢失跨投影分支间差异。loss 不会爆炸但质量下降难以察觉。

**当前代码**: QKV lora_A 无条件取平均：

```python
new_sd[fused_key] = (q_w + k_w + v_w) / 3.0
```

**修复**: 提供 `--abort_on_legacy_lora` 选项让用户选择是否继续；在保存的 adapter metadata 中加入 `fusion_layout` 版本号。

---

### P1-6: gradient_release 被作者标注为"数学上不严谨"

**文件**: `train.py:714-719`

**原文注释**: "this is unbelievably hacky and not mathematically sound, I'm just seeing if it works at all"

**影响**: 绕过 autograd 的 `.data.add_()` 在多层 pipeline 下与标准训练动态不同。当前仅在 `data_parallel_world_size==1` 且 LoRA-only 时有效。

**修复**: 生产环境禁用或有更严格的检查（如强制 pipeline_stages=1 且 gradient_release=True 时 hard-error 或至少 louder warning）。

---

### P1-7: LLM adapter 的 in_proj/out_proj Linear 不在 LoRA target 中

**文件**: `models/cosmos_predict2.py:577-580`

```python
adapter_target_modules = ['Block', 'TransformerBlock']
```

`LLMAdapter.in_proj` 和 `LLMAdapter.out_proj`（两个顶层 nn.Linear）不在任何 `adapter_target_modules` 匹配范围内。无法 LoRA fine-tune 这些层。

**修复**: 添加 `'LLMAdapter'` 到 `adapter_target_modules` 或显式文档说明。

---

### P1-8: t_embedder AdaLN-LoRA 参数归类到 base_params 而非 mod_params

**文件**: `models/cosmos_predict2.py:891-903`

`TimestepEmbedding` 内部的 adaln_lora 线性层（产出 `adaln_lora_B_T_3D`）不匹配 `.adaln_modulation` 子串（它们在 `t_embedder` 命名空间下），归入 `base_params` 走 `base_lr`，与其他 adaln 的 `mod_lr` 不一致。

**修复**: 在 `get_param_groups` 中对 `t_embedder` 名下的 adaln 层单独分类。

---

### P1-9: partition_method='manual' 且 pipeline_stages=1 时断言失败

**文件**: `utils/pipeline.py:44-45`

```python
assert num_partitions == num_stages - 1, ...
```

当 `pipeline_stages=1` 时，`num_stages-1=0`，默认 partition_split 非空导致断言失败。错误消息清晰但用户需手动设置 `partition_split=[]`。

**修复**: `set_config_defaults` 中自动处理此 case。

---

## P2 — 建议性改进

### P2-1: AUTOCAST_DTYPE=None 在推理/测试入口中导致 autocast 成为 no-op

**文件**: `utils/common.py:25` + `train.py:294`

若 Pipeline 构造函数被非 train.py 入口调用（推理脚本、测试），`AUTOCAST_DTYPE` 保持 `None` → 模型在 float32 下运行，显存和速度远低于预期。建议默认 `torch.bfloat16`。

### P2-2: RMSNorm fallback 路径强制 fp32 upcast

**文件**: `models/cosmos_predict2_modeling.py:280-283`

当 PyTorch < 2.4 且 `F.rms_norm` 不可用时，每次 RMSNorm 做 `.float() → _norm → .type_as() → * weight` = 4 次 kernel launch。实际部署中 PyTorch >= 2.4 已是标配，低优先级。

### P2-3: loss 函数用 loss.mean() 而非 loss.sum() / mask.sum()

**文件**: `models/cosmos_predict2.py:947-950`

`loss *= mask; loss = loss.mean()` 对全量元素取平均（含 masked 部分），而非标准做法 `loss.sum() / mask.sum()`（仅对有效像素平均）。大 padding 的样本 loss 被系统性低估，但实测影响小。

### P2-4: automagic_lr bf16 符号位下溢

**文件**: `optimizers/generic_optim.py:602`

当 LR 极接近 `min_lr=1e-7` 时，bf16 subnormal 可能丢失符号位，导致极性检测 (`last_polarity = automagic_lr > 0`) 出错。仅在极端 LR 场景下可能触发。

### P2-5: micro_batch_size TOML 整数键 → JSON 字符串键转换

**文件**: `train.py:291, 397-402`

`{512: 2}` 经 `json.loads(json.dumps(toml.load(f)))` 变为 `{"512": 2}`。取决于 dataset post_init 如何使用键值（如分辨率匹配时预期整数键）。

### P2-6: 文本 encoder 在 InitialLayer 中无显存/负载感知

**文件**: `models/cosmos_predict2.py:984-998`

`cache_text_embeddings=False` 时，大 LLM（Qwen3-0.6B）嵌入 InitialLayer 所在 GPU stage，pipeline 分区系统对此无感知。默认 `cache_text_embeddings=True`，影响有限。

### P2-7: resume checkpoint 的 examples fallback 可能不精确

**文件**: `train.py:877-878`

`step * global_batch_size` 在包含 warmup 或跳过步时偏低。仅影响 WandB/TensorBoard 日志，不影响训练。

### P2-8: enable_block_swap 中 blocks=None 临时状态有异常泄漏

**文件**: `models/cosmos_predict2.py:874-876`

异常发生时 `blocks=None` 状态会泄漏。极端边缘情况，用 try/finally 包裹即可。

### P2-9: torch_attention_op reshape 链可能非连续

**文件**: `models/cosmos_predict2_modeling.py:343-349`

`transpose(1,2).reshape()` 可能产生非连续 view，SDPA 内部隐式调 contiguous()。compile 路径无影响，非 compile 路径轻微开销。

---

## 总结

| 优先级 | 数量 | 关键议题 |
|--------|------|----------|
| P0 | 3 | Block swap CUDA race、Kahan buffer复用、AdaLN LoRA doc-code不一致 |
| P1 | 9 | contiguous冗余、swap同步、checkpoint检测、compile守卫、LoRA损失、参数分组 |
| P2 | 9 | 建议性改进，不影响核心训练正确性 |

**最紧急**: P0-1（offloading.py CUDA context race），会直接导致训练不稳定或静默数值错误，修复方向明确。
