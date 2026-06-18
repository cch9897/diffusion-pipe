# Expert-5: P1 架构修复方案

> **角色**: P1修复架构专家
> **日期**: 2026-06-18
> **范围**: 合并 CONSOLIDATED_REVIEW.md + anima_ft_review.md + anima_ft_review_2025-06-18.md 的 P1 清单
> **包括**: Expert-1 从 P0 降级的 3 项 (block swap CUDA context race, Kahan p.grad 复用, AdaLN doc-code不一致)
> **总项数**: 16 项独立 P1 问题 (去重后)

---

## 合并后的 P1 清单 (去重)

| # | 统一ID | 问题 | 源文档 | 分类 |
|---|--------|------|--------|------|
| P1-01 | ADALN_DIAG | AdaLN `adaln_modulation_2` 块对角结构训练中被破坏 | CONSOL-H1 | 正确性 |
| P1-02 | KAHAN_GRAD | Kahan 补偿求和中 p.grad 被就地覆写为临时 buffer | Expert-1降级 | 正确性 |
| P1-03 | ADALN_DOC | AdaLN LoRA remap docstring 承诺 cat 但代码无条件 average | Expert-1降级 | 正确性 |
| P1-04 | GRADREL_LR | gradient_release 学习率未按 GAS 缩放 | anima_ft-P1-3 | 正确性 |
| P1-05 | GRADREL_ADD | gradient_release `add_` 全局 monkeypatch | CONSOL-H9 / anima_ft-P1-6 | 正确性 |
| P1-06 | EVAL_SWAP | evaluate 异常时 block swap 状态泄漏 | anima_ft-P1-1 | 正确性 |
| P1-07 | T_EMBED_LR | t_embedder AdaLN-LoRA 参数归类到 base_params | anima_2025-P1-8 | 正确性 |
| P1-08 | SWAP_CTX | Block swap ThreadPool CUDA context race | Expert-1降级 | 资源安全 |
| P1-09 | SWAP_SYNC | Block swap 内 full synchronize 阻塞 pipeline | anima_ft-P1-6 / anima_2025-P1-2 | 资源/性能 |
| P1-10 | KAHAN_PCIE | Kahan CPU offload PCIe 双向同步 (non_blocking 退化) | anima_ft-P1-5 | 性能 |
| P1-11 | ATTN_BACKEND | Attention backend 硬编码为 'torch' | anima_ft-P1-2 | 性能 |
| P1-12 | COMPILE_DYN | torch_compile_dynamic 默认 False | anima_ft-P1-4 | 性能 |
| P1-13 | PER_PARAM | Per-param optimizer + foreach=False 大量小 kernel | anima_ft-P1-7 | 性能 |
| P1-14 | COMPILE_BS | blocks_to_swap 与 torch.compile recompile 冲突 | anima_ft-P1-8 / anima_2025-P1-4 | 性能/互斥 |
| P1-15 | CONTIG_280 | make_contiguous 280次/step 冗余拷贝 | anima_2025-P1-1 | 性能 |
| P1-16 | REENTRANT | torch.is_grad_enabled() 检测 reentrant checkpoint 不可靠 | anima_2025-P1-3 | 性能/正确性 |

> **已修复/已有方案排除项**: H3(CUDA Stream复用)✅ 已实现 `_swap_stream`、H4(CUDA Event复用)✅ 已实现 hasattr 守卫、H5(has_inf_or_nan)✅ 已改用 `dtype=torch.float32`、H6(circular ref)✅ 已添加 visited set、H7(mp.Queue) 需重新设计已暂缓、H8(model offload) 含 TODO 待验证。

---

## 逐项修复方案

---

### P1-01: ADALN_DIAG — AdaLN 块对角结构训练中被破坏

**问题描述**:
`adaln_modulation_2` 是普通 `nn.Linear(3R, 9D)`，初始化为块对角但训练使非对角块积累非零梯度。保存时 `_split_state_dict_keys` 仅提取对角块 → 非对角块学习到的信息被静默丢弃 → save/load 往返后行为不一致。

**影响范围**: `use_adaln_lora=True` 训练路径。`use_adaln_lora=False` 不受影响。

**位置**: `models/cosmos_predict2_modeling.py:1048-1050`, `models/cosmos_predict2.py:183-185`

---
#### [FIX_TACTICAL] 梯度掩码 (最小改动)

在 `adaln_modulation_2.weight.grad` 上注册 backward hook，将非对角块梯度清零：

```python
# 在 Block.__init__ 中 adaln_modulation_2 创建后
def _zero_off_diagonal_grad(grad):
    """Mask off-diagonal blocks to zero, enforcing block-diagonal structure."""
    D = self.x_dim
    R = self.adaln_lora_dim
    for i in range(3):
        for j in range(3):
            if i != j:
                grad[i*3*D:(i+1)*3*D, j*R:(j+1)*R] = 0
    return grad
self.adaln_modulation_2.weight.register_hook(_zero_off_diagonal_grad)
```

**改动量**: ~15 行
**风险**: 低。仅影响梯度，不影响前向。非对角块永远不收梯度，与 `_split_state_dict_keys` 的提取逻辑一致。
**缺点**: 浪费 8/9 的参数显存（非对角块永不被训练），但有效 rank 不变。训练时非对角块仍参与前向（虽然值为零），有微小计算开销。

---
#### [FIX_STRATEGIC] 三个独立 nn.Linear (根本解决)

将 `nn.Linear(3R, 9D)` 拆为 3 个独立的 `nn.Linear(R, 3D)`，从参数化层面消除非对角块：

```python
# __init__ 中替换
self.adaln_modulation_2_self_attn = nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False)
self.adaln_modulation_2_cross_attn = nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False)
self.adaln_modulation_2_mlp = nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False)

# forward 中
mid = self.adaln_modulation_1(silu_emb)  # (B, T, 3R)
R = self.adaln_lora_dim
br_self = self.adaln_modulation_2_self_attn(mid[..., :R])      # (B, T, 3D)
br_cross = self.adaln_modulation_2_cross_attn(mid[..., R:2*R]) # (B, T, 3D)
br_mlp = self.adaln_modulation_2_mlp(mid[..., 2*R:])           # (B, T, 3D)
```

**改动量**: ~80 行 (覆盖 `__init__`, `forward`, `_remap_state_dict_keys`, `_split_state_dict_keys`, `_remap_lora_keys`, `reset_parameters`, checkpoint 迁移逻辑)
**风险**: 中。需同步修改所有 state_dict 键映射逻辑，且需处理旧 checkpoint 兼容（检测旧格式 `adaln_modulation_2.weight` → 拆为 3 个对角块）。
**优点**: 零浪费、前向不需要无效计算、类型安全（不可能产生非对角梯度）。

---
#### [RECOMMENDATION] 战术修复

**理由**:
1. 改动量小 (~15行 vs ~80行)，风险极低
2. 战略修复涉及 LoRA remap、checkpoint 迁移等多处联动，需独立分支 + 完整往返测试
3. 战术修复的显存浪费可接受 (9D×3R 中 6/9 被浪费，但 LoRA rank 通常很小，如 R=256, D=3072 → 浪费 ~9 MB/block × 28 blocks ≈ 252 MB，占总模型 <1%)
4. 战略修复留到 v2 重构时一并处理

**依赖**: 无。独立修复。
**预估难度**: 战术 ★☆☆☆☆ (15min) | 战略 ★★★★★ (4h+)

---

### P1-02: KAHAN_GRAD — Kahan 求和中 p.grad 被就地覆写

**问题描述**:
`generic_optim.py:494` 行 `p.grad.copy_(p.detach())` 将梯度 tensor 覆写为参数当前值。`step()` 到 `zero_grad()` 之间任何读取 `p.grad` 的外部代码（hook、logger、gradient clipping）会拿到参数值而非梯度。同时 `p.grad` 作为临时 buffer 复用，语义混乱。

**位置**: `optimizers/generic_optim.py:490-498`, `optimizers/automagic.py:317-322`

---
#### [FIX_TACTICAL] 用独立临时 buffer 替代 p.grad

```python
# Before (generic_optim.py:494-496):
p.grad.copy_(p.detach())           # reuse p.grad as temp buffer
p.add_(shift)
shift.add_(p.grad.sub_(p))

# After:
tmp = torch.empty_like(p)
tmp.copy_(p.detach())              # independent temp buffer
p.add_(shift)
shift.add_(tmp.sub_(p))            # uses tmp, not p.grad
```

同样修复 `automagic.py:319-322` (该处用 `update` 变量而非 `p.grad`，但语义类似)。

**改动量**: ~6 行 × 2 处
**风险**: 极低。仅替换临时 buffer 来源，数学等价。
**显存开销**: `torch.empty_like(p)` 每个参数额外分配 ~= param size。各参数依次处理（不在循环内积压），峰值增量 ~= 最大单参数 size，通常 < 10 MB。

---
#### [FIX_STRATEGIC] Neumaier 无临时 buffer 实现

使用标准 Neumaier 补偿求和公式，完全消除临时 buffer 需求：

```python
# Neumaier compensated summation (no temp buffer):
# t = p + update; c = (t - p) - update; p = t (all via .data to avoid autograd)
y = update - shift          # y = correction + update
t = p.detach() + y          # t = p + y
new_shift = (t - p.detach()) - y  # compensation
p.copy_(t)                  # p = t
shift.copy_(new_shift)      # store compensation
```

**改动量**: ~10 行
**风险**: 低。Neumaier 是标准算法，已验证正确性（CONSOLIDATED_REVIEW B1 修复已确认此公式）。但 `p.copy_()` 是否会触发 autograd in-place 检测取决于上下文，需验证。

---
#### [RECOMMENDATION] 战术修复

**理由**:
1. 战术修复更安全——保留现有已验证的 "opposite-sign convention" 算法逻辑，仅替换临时 buffer 来源
2. 战略修复 (Neumaier) 虽更优雅，但 `p.copy_(t)` 在某些 PyTorch 版本/autograd 状态下可能触发 in-place 错误；现有 add_/sub_ 模式经过实战验证
3. 但长期应迁移到 Neumaier（已有 B1 修复为 Neumaier 建立正确性）

**依赖**: 必须在 B1 (Kahan 公式修复) 之后进行，因为 B1 修复了算法正确性，P1-02 在其基础上改善内存语义。
**预估难度**: ★★☆☆☆ (30min)

---

### P1-03: ADALN_DOC — AdaLN LoRA remap docstring 与代码不一致

**问题描述**:
`_remap_lora_keys` 的 docstring (line 218-222) 声明: "if the target rank < 3R, we average; if equal, we cat"，但代码 line 320-321 对所有情况无条件取平均。用户若提供 rank=3R 的融合后 LoRA 权重，会触发静默损毁 (应该无损 cat，实际做了有损 average)。

**位置**: `models/cosmos_predict2.py:218-328`

---
#### [FIX_TACTICAL] 更新 docstring，移除虚假承诺

```python
# Before (line 220-222):
# which requires the PEFT config to use rank 3R for these modules. Since
# adaln modulation typically uses a small rank, this is handled by padding
# lora_A to match: if the target rank < 3R, we average; if equal, we cat.

# After:
# This remapping is always lossy (3R -> R via averaging). If you need
# lossless round-trip, retrain with the fused adaln_modulation_1/2 modules.
# There is no cat path even when rank == 3R because the fused module's
# lora_A shape (R, 3*adaln_lora_dim) differs in semantic layout from
# the legacy 3×(R, adaln_lora_dim) split.
```

**改动量**: ~5 行
**风险**: 无。

---
#### [FIX_STRATEGIC] 实现 rank 检测 + cat 路径

```python
if lora_type == 'lora_A':
    # Detect if we can losslessly cat (rank == 3R after fusion)
    fused_rank = l1_list[0].shape[0]  # R
    per_branch_dim = l1_list[0].shape[1]  # adaln_lora_dim
    if fused_rank == 3 * per_branch_dim:
        # rank == 3*adaln_lora_dim: can cat losslessly
        new_sd[fused_key_1] = torch.cat(l1_list, dim=0)  # (3R, ald)
        new_sd[fused_key_2] = torch.cat(l2_list, dim=0)  # (3R, ald)
    else:
        warnings.warn(...)
        new_sd[fused_key_1] = sum(l1_list) / 3.0
        new_sd[fused_key_2] = sum(l2_list) / 3.0
```

**改动量**: ~20 行
**风险**: 低。增加了无损路径，不破坏现有行为（现有行为 = always average = 走 else 分支）。

---
#### [RECOMMENDATION] 战略修复（两者都做）

**理由**:
1. 战术修复（更新 docstring）零风险，立即执行
2. 战略修复的 cat 路径提供无损迁移能力，对未来用户有价值
3. 两者互补：docstring 更新说明默认行为 + cat 路径处理特殊情况

**依赖**: 无。
**预估难度**: ★☆☆☆☆ (30min)

---

### P1-04: GRADREL_LR — gradient_release 学习率未按 GAS 缩放

**问题描述**:
`train.py:727-732` 对 betas 做 `**(1/gas)` 缩放以补偿更频繁的更新，但 LR 未除以 GAS。每 micro-batch 执行完整 `optimizer.step()`，等效更新频率提高 GAS 倍。Adam 的 effective step size ≈ lr / (1 - β₁)，betas 缩放部分补偿但不等价。

**位置**: `train.py:727-748`

---
#### [FIX_TACTICAL] 添加 `gas_lr_scale` 配置项

```python
gas = ds_config['gradient_accumulation_steps']
# Allow user to explicitly scale LR per micro-batch step
lr_scale = optim_config.get('gas_lr_scale', 1.0 / gas)  # default: scale by 1/gas
if 'lr' in kwargs:
    kwargs['lr'] = kwargs['lr'] * lr_scale
```

同时在代码注释/文档中说明默认行为。保守用户设 `gas_lr_scale=1.0` 维持现有行为。

**改动量**: ~5 行
**风险**: 极低。默认行为改变需文档化。

---
#### [FIX_STRATEGIC] 累积梯度模式替代 per-step

修改 gradient_release 为真正的 "累积梯度 → 定期 step" 模式，而非每 micro-batch step：

```python
# 伪代码
accumulated_grads = []
for micro_batch in micro_batches:
    loss = forward_backward(micro_batch)
    accumulated_grads.append(capture_grads())
    if len(accumulated_grads) >= gas:
        merged_grads = merge_gradients(accumulated_grads)
        optimizer.step(merged_grads)
        optimizer.zero_grad()
```

**改动量**: ~200+ 行（需重新设计梯度捕获和合并逻辑）
**风险**: 高。与 DeepSpeed pipeline engine 的集成复杂度高，`register_post_accumulate_grad_hook` 可能无法正确工作。

---
#### [RECOMMENDATION] 战术修复

**理由**:
1. gradient_release 本身就是实验性功能（代码注释自认 "unbelievably hacky"），不值得投入大量工程资源做战略重构
2. 添加 `gas_lr_scale` 配置项成本极低，给用户提供控制权
3. 长期应推进到 P1-05 的战略方案（消除 gradient_release）

**依赖**: 与 P1-05 (GRADREL_ADD) 共享 root cause，建议一起处理。
**预估难度**: ★☆☆☆☆ (15min)

---

### P1-05: GRADREL_ADD — gradient_release `add_` 全局 monkeypatch

**问题描述**:
`train.py:716-719` 全局替换所有可训练参数的 `add_` 方法为 `self.data.add_()`，绕过 autograd in-place 检测。注释自认 "unbelievably hacky and not mathematically sound"。仅 gradient_release 模式触发，但一旦激活全局生效。非梯度释放路径的 `p.add_()` 调用会静默破坏计算图。

**位置**: `train.py:713-719`

---
#### [FIX_TACTICAL] 隔离 monkeypatch 作用域 + 强化守卫

```python
# Before: monkeypatch applied unconditionally to all parameters
for p in model_parameters:
    p.add_ = add_.__get__(p)

# After: only apply when gradient_release=True AND pipeline_stages=1
# Also guard data_parallel_world_size (existing assert is in _exec_reduce_grads)
if ds_config.get('pipeline', {}).get('stages', 1) > 1:
    raise ValueError(
        'gradient_release is incompatible with pipeline_stages > 1. '
        'The add_ monkeypatch would silently corrupt autograd state '
        'across pipeline stages.'
    )
for p in model_parameters:
    p.add_ = add_.__get__(p)
```

同时在代码顶部加详细注释（如 FIX_PLAN.md Phase 3 所建议）。

**改动量**: ~10 行
**风险**: 低。仅增强守卫，不改变行为。

---
#### [FIX_STRATEGIC] 用 `register_post_accumulate_grad_hook` 替代 monkeypatch

当前 gradient_release 的架构问题是：在 pipeline parallel 下，`p.add_()` 的 monkeypatch 使不同 stage 看到不同版本的参数。正确方案：

```python
# 不使用 add_ monkeypatch
# 而是利用 register_post_accumulate_grad_hook 在每 micro-batch 后执行 optimizer step
# 关键：确保所有 stages 的 hook 在相同 micro-batch 上触发

def _gradient_release_step(p):
    """Per-parameter optimizer step after each micro-batch's grad accumulation."""
    if p.grad is not None:
        optimizer_dict[p].step()
        optimizer_dict[p].zero_grad()

for p in model_parameters:
    p.register_post_accumulate_grad_hook(lambda p=p: _gradient_release_step(p))
```

**改动量**: ~50 行
**风险**: 中。需验证 DeepSpeed pipeline engine 的 `post_accumulate_grad_hook` 在 pipeline parallel 下的触发时序。

---
#### [RECOMMENDATION] 战术修复（短期）+ 长期禁用计划

**理由**:
1. gradient_release 功能本身标记为实验性，战略重构不值得
2. 战术修复加硬守卫防止危险组合，成本极低
3. 长期建议：如果 gradient_release 在 LoRA 训练中确实有显存优势，应将其重构为 DeepSpeed 原生梯度累积模式；否则移除该功能

**依赖**: 无（独立）。
**预估难度**: ★☆☆☆☆ (15min 战术) | ★★★★☆ (4h+ 战略)

---

### P1-06: EVAL_SWAP — evaluate 异常时 block swap 状态泄漏

**问题描述**:
`train.py:237-249` 的 `evaluate()` 函数调用 `prepare_block_swap_inference()` (设置 `forward_only=True`) 和随后的 `prepare_block_swap_training()` 不在 try/finally 中。评估异常后模型永久卡在 `forward_only=True` → backward hook 跳过 block swap → 被换出 block 的梯度丢失 → 训练静默损坏。

**位置**: `train.py:237-249`

---
#### [FIX_TACTICAL] try/finally 包围

```python
def evaluate(model, model_engine, eval_dataloaders, tb_writer, step,
             eval_gradient_accumulation_steps, disable_block_swap):
    if len(eval_dataloaders) == 0:
        return
    empty_cuda_cache()
    model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)
    try:
        with torch.no_grad(), isolate_rng():
            seed = get_rank()
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps)
    finally:
        empty_cuda_cache()
        model.prepare_block_swap_training()
```

**改动量**: ~5 行 (重新缩进)
**风险**: 极低。

---
#### [FIX_STRATEGIC] Context manager 封装

```python
@contextmanager
def block_swap_inference_mode(model, disable_block_swap=False):
    """Context manager that safely enters/exits block swap inference mode."""
    model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)
    try:
        yield
    finally:
        model.prepare_block_swap_training()

# 使用
with block_swap_inference_mode(model, disable_block_swap_for_eval):
    with torch.no_grad(), isolate_rng():
        _evaluate(...)
```

**改动量**: ~15 行
**风险**: 极低。

---
#### [RECOMMENDATION] 战略修复 (context manager)

**理由**:
1. Context manager 是 Python 标准模式，比 try/finally 更可读
2. 可复用：任何需要临时切换 block swap 模式的场景都可以使用
3. 改动量极小

**依赖**: 无。
**预估难度**: ★☆☆☆☆ (10min)

---

### P1-07: T_EMBED_LR — t_embedder AdaLN-LoRA 参数归类错误

**问题描述**:
`get_param_groups()` 中 `TimeestepEmbedding` 内部的 adaln_lora 线性层（产出 `adaln_lora_B_T_3D`）命名空间为 `t_embedder.1.adaln_modulation_1`，不匹配 `.adaln_modulation` 子串（因为 `t_embedder` 是 `nn.Sequential`，其子模块名被包装）。这些参数归入 `base_params` 走 `base_lr`，与其他 adaln 的 `mod_lr` 不一致。

**位置**: `models/cosmos_predict2.py:891-906`

---
#### [FIX_TACTICAL] 扩展子串匹配

```python
# Before:
elif '.adaln_modulation' in name:
    mod_params.append(p)

# After:
elif '.adaln_modulation' in name or ('t_embedder' in name and 'adaln' in name):
    mod_params.append(p)
```

**改动量**: 1 行
**风险**: 低。但匹配 `t_embedder` + `adaln` 可能过于宽泛。

---
#### [FIX_STRATEGIC] 基于模块类型分类

```python
def _classify_param(name, module):
    """Classify parameter by module type, not name pattern."""
    if isinstance(module, LLMAdapter):
        return 'llm_adapter'
    if isinstance(module, Attention):
        if module.is_selfattn:
            return 'self_attn'
        else:
            return 'cross_attn'
    if isinstance(module, GPT2FeedForward):
        return 'mlp'
    if hasattr(module, 'is_adaln_modulation') and module.is_adaln_modulation:
        return 'mod'
    return 'base'

# 在各 AdaLN 相关模块中设置标记
class Block:
    def __init__(self):
        self.adaln_modulation_1.is_adaln_modulation = True
        self.adaln_modulation_2.is_adaln_modulation = True
```

**改动量**: ~40 行
**风险**: 中。需要为所有 AdaLN 相关模块标记。但避免了字符串匹配的脆弱性。

---
#### [RECOMMENDATION] 战术修复

**理由**:
1. 改动量极小 (1 行)，风险可控
2. 字符串匹配虽脆弱但可测试覆盖
3. 战略修复的模块类型方案是更好的长期设计，但需跨多文件协调，投入产出比低

**依赖**: 无。
**预估难度**: ★☆☆☆☆ (5min)

---

### P1-08: SWAP_CTX — Block swap ThreadPool CUDA context race

**问题描述**:
`offloading.py:144,171` 使用 `ThreadPoolExecutor(max_workers=1)` 在子线程中执行 CUDA 操作 (`swap_weight_devices_cuda`)。CUDA context 是 thread-local 的，子线程中的 `stream.synchronize()`、`wait_stream()` 等 API 调用可能因缺少 context 成为 no-op，导致主线程在 H2D 完成前开始计算 → illegal loss value 或静默数值错误。

**Expert-1 降级说明**: 从 P0 降为 P1 因为 (a) max_workers=1 限制并发度，(b) 实际测试中未观察到 crash，(c) PyTorch 2.x 的 CUDA context 管理比预期更宽容。但风险仍然是真实存在的——CUDA 规范不保证跨线程 context 共享。

**位置**: `utils/offloading.py:144,171-173`

---
#### [FIX_TACTICAL] 移除 ThreadPoolExecutor，改为主线程异步 CUDA stream

```python
# Before:
self.thread_pool = ThreadPoolExecutor(max_workers=1)

def _submit_move_blocks(self, ...):
    self.futures[block_idx_to_cuda] = self.thread_pool.submit(
        move_blocks, ...
    )

def _wait_blocks_move(self, block_idx):
    future = self.futures.pop(block_idx)
    _, bidx_to_cuda = future.result()

# After (no ThreadPoolExecutor):
def _submit_move_blocks(self, ...):
    # Submit D2H/H2D to dedicated CUDA stream (already exists as _swap_stream)
    self.swap_weight_devices(block_to_cpu, block_to_cuda)
    self.futures[block_idx_to_cuda] = block_idx_to_cuda  # marks "done"

def _wait_blocks_move(self, block_idx):
    # Synchronize the swap stream with current stream
    torch.cuda.current_stream().wait_stream(_swap_stream)
    # (swap already completed synchronously above)
    result = self.futures.pop(block_idx)
```

实际上，由于当前 `swap_weight_devices_cuda` 已经使用 module-level `_swap_stream` 且在 `_submit_move_blocks` 中同步调用（ThreadPool 并未真正提供异步收益——因为 `_wait_blocks_move` 立即 `.result()` 等待），最简单的修复就是：

```python
# 删除 ThreadPoolExecutor，直接在主线程调用 swap_weight_devices
# 将 _submit_move_blocks 改为同步调用
def _submit_move_blocks(self, block_idx_to_cpu, block_idx_to_cuda):
    self.swap_weight_devices(
        self.blocks[block_idx_to_cpu],
        self.blocks[block_idx_to_cuda]
    )
    self.futures[block_idx_to_cuda] = True

def _wait_blocks_move(self, block_idx):
    assert self.futures.pop(block_idx, None) is not None
```

**改动量**: ~10 行（删除 ThreadPoolExecutor，简化 _submit/_wait）
**风险**: 极低。ThreadPool 本来就是 max_workers=1 的同步等效——backward hook 中 `_submit_move_blocks(swap)` 后下一个 hook 立即 `_wait_blocks_move(wait)`，所以 ThreadPool 没有提供实际并行。

---
#### [FIX_STRATEGIC] 真正的异步 block swap pipeline

使用专用 CUDA stream 实现真正的计算/Swap 重叠：

```python
# 用 CUDA event 而非 ThreadPool 实现异步
self.swap_stream = torch.cuda.Stream()
self.swap_events = {}  # block_idx -> CUDA event

def _submit_move_blocks(self, ...):
    with torch.cuda.stream(self.swap_stream):
        self.swap_weight_devices(block_to_cpu, block_to_cuda)
    event = torch.cuda.Event()
    event.record(self.swap_stream)
    self.swap_events[block_idx_to_cuda] = event

def _wait_blocks_move(self, block_idx):
    event = self.swap_events.pop(block_idx)
    torch.cuda.current_stream().wait_event(event)
```

但这需要确保 `swap_weight_devices_cuda` 内部不使用 `current_stream().synchronize()`（目前 line 74 有全同步），且 `wait_stream(stream)` (line 94) 改为 event-based。

**改动量**: ~40 行
**风险**: 中。需要重组 swap_weight_devices_cuda 内部的同步逻辑。

---
#### [RECOMMENDATION] 战术修复

**理由**:
1. ThreadPoolExecutor 在当前架构下没有提供实际并行收益（max_workers=1 + 立即 wait）
2. 移除 ThreadPoolExecutor 消除 CUDA context race 风险，且不损失性能
3. 战略修复的异步 pipeline 可后续在 block swap 重构时实现（与 P1-09 联动）

**依赖**: 可与 P1-09 (SWAP_SYNC) 合并处理。
**预估难度**: ★★☆☆☆ (30min)

---

### P1-09: SWAP_SYNC — block swap 冗余 synchronize 阻塞 pipeline

**问题描述**:
`offloading.py:74` 在每次权重交换前执行 `torch.cuda.current_stream().synchronize()` 全同步，stall 主计算流。注释说明 "this prevents the illegal loss value"，但这意味着所有待处理的 CUDA 操作都必须完成，增加 1-3ms/swap × 2×blocks_to_swap/step。此外，`swap_weight_devices_cuda` 内部还有 `stream.synchronize()` + `current_stream().wait_stream(stream)` (line 93-94)，双重同步。

**位置**: `utils/offloading.py:74, 93-94`

---
#### [FIX_TACTICAL] 用独立 CUDA stream event 替代 full synchronize

```python
# Before (line 74):
torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value
stream = _swap_stream
with torch.cuda.stream(stream):
    # ... D2H + H2D ...
stream.synchronize()
torch.cuda.current_stream().wait_stream(stream)

# After: 移除 line 74 的 full sync，仅依赖 swap stream 的同步
# 用 event 替代 stream.synchronize() 以便后续异步等待
global _swap_stream
if _swap_stream is None:
    _swap_stream = torch.cuda.Stream()
stream = _swap_stream
with torch.cuda.stream(stream):
    # ... D2H + H2D ...
# Record event on swap stream
swap_done = torch.cuda.Event()
swap_done.record(stream)
torch.cuda.current_stream().wait_event(swap_done)  # 仅等待 swap stream
```

**改动量**: ~5 行
**风险**: 低。event-based 同步比 full synchronize 更细粒度，仅等待 swap stream 而非所有 GPU 操作。

---
#### [FIX_STRATEGIC] pipeline 化 block swap

结合 P1-08 的战略方案：D2H 和 H2D 分两步，利用 swap stream 与计算 stream 真正重叠：

```python
# 前向中: 提交 D2H (非阻塞)
with torch.cuda.stream(self.swap_stream):
    for job in d2h_jobs:
        cpu_tensor = cuda_tensor.to('cpu', non_blocking=True)
    self.d2h_event = torch.cuda.Event()
    self.d2h_event.record(self.swap_stream)

# 计算继续进行 (与 D2H 重叠)...

# Backward hook 中: 等待 D2H 完成，提交 H2D
self.d2h_event.wait(self.swap_stream)
with torch.cuda.stream(self.swap_stream):
    for job in h2d_jobs:
        cuda_tensor.copy_(cpu_tensor, non_blocking=True)
    self.h2d_event = torch.cuda.Event()
    self.h2d_event.record(self.swap_stream)

# 使用 block 前: 
torch.cuda.current_stream().wait_event(self.h2d_event)
```

但这需要重构 `swap_weight_devices_cuda` 的整体结构。

**改动量**: ~60 行
**风险**: 中-高。需确保 D2H/H2D 分步后的正确性。

---
#### [RECOMMENDATION] 战术修复

**理由**:
1. 移除 line 74 full synchronize 并用 event 替代是最小改动
2. Event-based 方案保留了同步安全性（防止 illegal loss value）同时减少了不必要的全 GPU 同步
3. 战略 pipeline 化需要大量重构，留到 block swap v2

**依赖**: 与 P1-08 (SWAP_CTX) 合并处理效果更佳。
**预估难度**: ★★☆☆☆ (45min)

---

### P1-10: KAHAN_PCIE — Kahan CPU offload PCIe 双向同步

**问题描述**:
`generic_optim.py:492,498` 当 `kahan_buffer_device='cpu'` 时，每步做全参数 PCIe round-trip。H2D 用 `non_blocking=True` 但 CPU 内存未 pin → 退化为同步传输。7B 模型约 14 GB/step PCIe 传输。同时 line 497-498: `state['shift'].to(kahan_buffer_device)` 的 `non_blocking=True` 在 checkpoint save 后会导致 CUDA error（已标注 TODO）。

**位置**: `optimizers/generic_optim.py:490-498`

---
#### [FIX_TACTICAL] 评估是否需要 CPU offload + 添加 pin_memory

```python
# 在 state 初始化时 pin shift memory
if 'shift' not in state:
    shift_cpu = torch.zeros_like(p)
    if kahan_buffer_device.type == 'cpu':
        shift_cpu = shift_cpu.pin_memory()  # 使 non_blocking 真正异步
    state['shift'] = shift_cpu

# 传输时:
shift = state['shift'].to(p.device, non_blocking=True)
# ...
# 传回:
if kahan_buffer_device.type == 'cpu':
    state['shift'] = shift.to(kahan_buffer_device, non_blocking=True)
else:
    state['shift'] = shift.to(kahan_buffer_device)
```

**改动量**: ~10 行
**风险**: 低。Pin memory 增加系统内存占用但 shift buffer 通常很小。

---
#### [FIX_STRATEGIC] 保持 kahan buffer 在 GPU

默认 `kahan_buffer_device` 设为与参数相同设备（通常是 GPU），避免 PCIe 传输。仅当显存极度紧张时回退 CPU offload。

```python
# 默认: 保持 shift 在 GPU
kahan_buffer_device = optim_config.get('kahan_buffer_device', p.device)
# 仅在显存不足时允许 CPU offload
```

同时评估 GPU 上 kahan buffer 的显存开销（shift size = param size，对 LoRA 训练 < 100 MB）。

**改动量**: ~5 行
**风险**: 极低。

---
#### [RECOMMENDATION] 战略修复为主 + 战术修复为辅

**理由**:
1. 对于 LoRA 训练，kahan buffer 在 GPU 的显存开销可忽略（LoRA params < 100 MB）
2. 默认 GPU 驻留消除了 PCIe 传输问题
3. 对于全参数训练需要 CPU offload 的场景，pin_memory 修复仍有价值

**依赖**: 无。
**预估难度**: ★★☆☆☆ (30min)

---

### P1-11: ATTN_BACKEND — Attention backend 硬编码为 'torch'

**问题描述**:
`cosmos_predict2_modeling.py:1215` 硬编码 `atten_backend = 'torch'`，覆盖 `Attention.__init__` 默认的 `'transformer_engine'`。放弃 TE DotProductAttention (FlashAttention-2 fused kernel)，长序列 attention 慢 2-3x。TE 后端路径是死代码且缺少 import。

**位置**: `models/cosmos_predict2_modeling.py:1215, 424`

---
#### [FIX_TACTICAL] 添加 transformer_engine import + 可配置 backend

```python
# import 区:
try:
    from transformer_engine.pytorch import DotProductAttention
    _HAS_TE = True
except ImportError:
    DotProductAttention = None
    _HAS_TE = False

# MiniTrainDIT.__init__:
atten_backend = self.model_config.get(
    'attention_backend',
    'transformer_engine' if _HAS_TE else 'torch'
)
```

**改动量**: ~10 行
**风险**: 低。仅在 TE 可用时启用，否则回退 torch。

---
#### [FIX_STRATEGIC] Attention backend 统一抽象

```python
class AttentionBackend(enum.Enum):
    TORCH_SDPA = 'torch'
    TRANSFORMER_ENGINE = 'transformer_engine'
    FLASH_ATTN = 'flash_attn'  # 未来: flash-attn 库直接调用

def _create_attention_op(backend, **kwargs):
    if backend == AttentionBackend.TRANSFORMER_ENGINE:
        return DotProductAttention(**kwargs)
    elif backend == AttentionBackend.TORCH_SDPA:
        return functools.partial(F.scaled_dot_product_attention, **kwargs)
    # ...
```

**改动量**: ~60 行
**风险**: 中。需要抽象所有 attention 后端接口。

---
#### [RECOMMENDATION] 战术修复

**理由**:
1. 两行改动即可恢复 TE 后端，收益 15-30%
2. 前提: 确认训练环境安装了 transformer_engine（`import transformer_engine` 测试）
3. 战略抽象留到 attention 模块重构时

**依赖**: 需先在训练环境确认 TE 是否可用。
**预估难度**: ★★☆☆☆ (30min + 测试)

---

### P1-12: COMPILE_DYN — torch_compile_dynamic 默认 False

**问题描述**:
`cosmos_predict2.py:836` 中 `torch_compile_dynamic` 默认为 `False`。多 bucket (多分辨率) 训练时每个 (block, shape) 组合触发独立 recompile。28 blocks × N buckets = 28N 次编译，每次数十秒 + 显存膨胀。

**位置**: `models/cosmos_predict2.py:836`

---
#### [FIX_TACTICAL] 多 bucket 场景默认 True

```python
# Before:
compile_dynamic = self.model_config.get('torch_compile_dynamic', False)

# After: auto-detect multi-bucket and default to dynamic=True
has_multiple_buckets = len(self.model_config.get('size_buckets', [1])) > 1
compile_dynamic = self.model_config.get(
    'torch_compile_dynamic',
    has_multiple_buckets  # default True for multi-bucket, False for single-bucket
)
```

**改动量**: 3 行
**风险**: 低。`dynamic=True` 对本就是单 bucket 的场景也安全（只是编译略微保守）。已有 warning (line 846) 提示用户。

---
#### [FIX_STRATEGIC] 编译策略统一配置

```python
# 集中管理 compile 配置
compile_config = {
    'mode': config.get('torch_compile_mode', 'default'),
    'dynamic': config.get('torch_compile_dynamic', None),  # None = auto
    'fullgraph': False,
}
# auto-detect
if compile_config['dynamic'] is None:
    compile_config['dynamic'] = has_multiple_buckets or blocks_to_swap > 0
```

**改动量**: ~20 行
**风险**: 低。

---
#### [RECOMMENDATION] 战术修复

**理由**: 改动量极小，auto-detect 逻辑简单可靠。

**依赖**: 无。
**预估难度**: ★☆☆☆☆ (10min)

---

### P1-13: PER_PARAM — Per-param optimizer + foreach=False

**问题描述**:
`train.py:741-744` gradient_release 模式下为每个参数创建独立 optimizer 实例，并强制 `foreach=False`。数千小 kernel launch/step。对比 foreach=True 批量 kernel，kernel launch overhead 增加 5-10x。

**位置**: `train.py:721-722, 741, 744`

---
#### [FIX_TACTICAL] 按 param group 合并 optimizer

```python
# Before: per-param optimizer
for p in pg['params']:
    optimizer_dict[p] = klass([p], **param_kwargs)

# After: group params by shared config
grouped_params = {}
for p in pg['params']:
    lr_key = pg['lr']
    if lr_key not in grouped_params:
        grouped_params[lr_key] = []
    grouped_params[lr_key].append(p)

for lr_key, params in grouped_params.items():
    param_kwargs['lr'] = lr_key
    opt = klass(params, **param_kwargs)
    for p in params:
        optimizer_dict[p] = opt
```

同时启用 `foreach=True`（但需确认 gradient_release 模式下是否兼容）。

**改动量**: ~20 行
**风险**: 中。需要验证共享 optimizer 在 per-parameter hook 场景下的正确性。

---
#### [FIX_STRATEGIC] 消除 per-param optimizer 需求

gradient_release 的 per-param optimizer 源于每 micro-batch 的独立 `step()` + `zero_grad()`。如果采用真正的梯度累积（见 P1-04 战略方案），则可以使用单一共享 optimizer：

```python
optimizer = klass(all_params, foreach=True, **kwargs)
```

**改动量**: 与 P1-04 战略方案重叠
**风险**: 高。

---
#### [RECOMMENDATION] 战术修复

**理由**:
1. 按 param group 合并显著减少 optimizer 实例数（从 N_params → N_groups）
2. 不改变 gradient_release 的整体架构
3. 战略方案需要梯度累积重构，成本高

**依赖**: 仅在 gradient_release 模式生效。可与 P1-04/P1-05 一起处理。
**预估难度**: ★★☆☆☆ (1h)

---

### P1-14: COMPILE_BS — blocks_to_swap 与 torch.compile 冲突

**问题描述**:
Block swap 通过 `module.weight.data = ...` 原地修改触发 dynamo guard 失效 → 持续 recompile。代码已有 warning (line 841-842) 但未阻止。组合导致每步重新编译，训练几乎停滞。

**位置**: `models/cosmos_predict2.py:841-842`

---
#### [FIX_TACTICAL] 硬互斥检测

```python
if self.model_config.get('torch_compile', False) and self.config.get('blocks_to_swap', 0):
    raise ValueError(
        'torch_compile and blocks_to_swap are mutually incompatible. '
        'blocks_to_swap mutates module.weight.data on every step, '
        'which invalidates dynamo guards and triggers recompilation. '
        'Use one or the other, not both.'
    )
```

同样添加 unsloth + compile 互斥检测。

**改动量**: 5 行
**风险**: 无。防止用户误配置。

---
#### [FIX_STRATEGIC] 使 block swap 兼容 compile

通过 `torch._dynamo.mark_dynamic()` 标记 swap block 的权重 tensor，或使用 custom Triton kernel 替代 in-place weight swap：

```python
# 方案A: 标记权重为动态
for block in swapped_blocks:
    for name, p in block.named_parameters():
        torch._dynamo.mark_dynamic(p, 0)  # 在对应维度标记动态
```

但此方案在 PyTorch 2.x 中不可靠（dynamo guards 对 `.data` 修改敏感）。

**方案B**: 用 `torch.compile` 的 `dynamic=True` 缓解（但不根本解决）。

**方案C**: 修改 block swap 为 non-in-place 方式（交换整个模块引用而非 `.data`），使 compile 的 guard 不被触发。

**改动量**: ~100+ 行
**风险**: 高。需要深入理解 dynamo guard 机制。

---
#### [RECOMMENDATION] 战术修复

**理由**:
1. make compile + block_swap 互斥是安全的短期方案
2. 使二者兼容是 PyTorch 层面的难题（dynamo 不支持 `.data` in-place 修改），不值得投入
3. 用户可以在 compile 和 block_swap 之间选择：显存紧张 → block_swap (不 compile)；显存充裕 → compile (不 swap)

**依赖**: 无。
**预估难度**: ★☆☆☆☆ (10min)

---

### P1-15: CONTIG_280 — make_contiguous 冗余拷贝

**问题描述**:
`cosmos_predict2.py:1016` + `base.py:33-34` 中 `make_contiguous()` 对 block 间完全不变的 `t_embedding_B_T_D`、`crossattn_emb`、`rope_emb_L_1_1_D`、`adaln_lora_B_T_3D`、`timesteps_B_T` 每 step 做 28 blocks × 5 tensors = 140 次无意义 contiguous。~2-4% 吞吐损失。

**位置**: `models/cosmos_predict2.py:1016`, `models/base.py:33-34`

---
#### [FIX_TACTICAL] 仅对 x 做 contiguous

```python
# Before (在 InitialLayer.forward 末尾):
outputs = make_contiguous(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb,
                          t5_input_ids, attn_mask, t5_attn_mask,
                          rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T)

# After:
x_B_T_H_W_D = x_B_T_H_W_D.contiguous()
outputs = (x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb,
           t5_input_ids, attn_mask, t5_attn_mask,
           rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T)
```

同时修改 `TransformerLayer.forward` 末尾（line ~1050+）同样只对 x 做 contiguous。

**改动量**: ~4 行 × 2 处
**风险**: 低。需确认下游（下一 block 的第一层）不依赖其他 tensor 的 stride。t_embedding/crossattn_emb/rope_emb 等在 block 间完全不变（仅索引访问），stride 不影响正确性。

---
#### [FIX_STRATEGIC] 重构数据流：不变 tensor 不经过 block forward

```python
# 在 TransformerLayer.forward 中:
# 仅在 block 间传递 x 和 positional extras
# t_embedding/crossattn_emb/adaln_lora 由外部作用域直接引用
class TransformerLayer(nn.Module):
    def forward(self, inputs):
        x, shared_context = inputs  # shared_context 包含所有不变 tensor
        x = self.block(x, **shared_context)
        x = x.contiguous()
        return (x, shared_context)
```

**改动量**: ~50 行 (修改 TransformerLayer 的 forward 签名)
**风险**: 中。需要修改 pipeline 分区中的张量传递逻辑。

---
#### [RECOMMENDATION] 战术修复

**理由**:
1. 仅改 2 处 contiguous 调用，4 行改动，2-4% 吞吐提升
2. 不变 tensor 的 stride 确实不被下游依赖
3. 战略重构的数据流改进可留到 pipeline 架构优化时

**依赖**: 无。
**预估难度**: ★☆☆☆☆ (15min + 验证)

---

### P1-16: REENTRANT — is_grad_enabled() 检测 reentrant checkpoint 不可靠

**问题描述**:
`offloading.py:299, 313` 用 `torch.is_grad_enabled()` 检测 reentrant checkpoint 的第二遍 forward (replay)，以跳过 block swap。但 unsloth checkpoint 使用 `@torch.amp.custom_fwd`，不改变 `is_grad_enabled()` 状态 → replay forward 时也触发 block swap → D2H/H2D 与重计算竞态。注释已注明局限但未处理 unsloth 路径。

**位置**: `utils/offloading.py:299, 313`

---
#### [FIX_TACTICAL] 显式 flag 替代隐式检测

```python
# 在 ModelOffloader 或 Block 上添加：
self._is_replay_forward = False

# 在 checkpoint hook 中切换：
def checkpoint_hook(module, inputs):
    module._is_replay_forward = True
    outputs = original_forward(*inputs)
    module._is_replay_forward = False
    return outputs

# wait_for_block / submit_move_blocks_forward 中：
if self.reentrant_activation_checkpointing and self._is_replay_forward:
    return  # skip block swap in replay forward
```

**改动量**: ~15 行
**风险**: 低。显式 flag 比 `is_grad_enabled()` 更精确。

---
#### [FIX_STRATEGIC] 重构 checkpoint 集成

将 block swap 逻辑移到 checkpoint 外部，不依赖 forward pass 内的条件检测：

```python
# 在 Block.forward 外层（TransformerLayer.forward）中控制 swap
# 而非在 Block.forward 内部
class TransformerLayer:
    def forward(self, x, ...):
        if not is_replay:
            self.offloader.submit_move_blocks_forward(block_idx)
        if self.activation_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                self.block.forward, x, ..., use_reentrant=True
            )
        else:
            x = self.block.forward(x, ...)
        if not is_replay:
            self.offloader.wait_for_block(block_idx)
```

**改动量**: ~40 行
**风险**: 中。需要重组 forward 调用链。

---
#### [RECOMMENDATION] 战术修复

**理由**:
1. 显式 flag 是直接、可验证的修复
2. 无需重构 forward 调用链
3. 战略方案需要与 activation checkpointing 的统一重构一起做

**依赖**: 无。
**预估难度**: ★★☆☆☆ (1h)

---

## 依赖关系图

```
P1-02 (KAHAN_GRAD)
  └── depends on: B1 (Kahan公式修复, P0) must be done first
      (B1 fixes the math, P1-02 fixes the buffer semantics on top)

P1-08 (SWAP_CTX) ◄──► P1-09 (SWAP_SYNC)
  └── strongly coupled: both modify offloading.py's swap logic
  └── recommend merge into one PR

P1-04 (GRADREL_LR) ◄──► P1-05 (GRADREL_ADD) ◄──► P1-13 (PER_PARAM)
  └── all three are gradient_release concerns
  └── can be addressed independently but share context

P1-14 (COMPILE_BS) ── mutually exclusive guard, independent

P1-01 (ADALN_DIAG) ── independent (but related to P1-03)

P1-03 (ADALN_DOC) ── independent (but related to P1-01)

P1-06 (EVAL_SWAP) ── independent

P1-07 (T_EMBED_LR) ── independent

P1-10 (KAHAN_PCIE) ── independent (but related to P1-02)

P1-11 (ATTN_BACKEND) ── independent

P1-12 (COMPILE_DYN) ── independent

P1-15 (CONTIG_280) ── independent

P1-16 (REENTRANT) ── independent
```

---

## 修复路线图 (分阶段)

### Phase 0: P0 前置 (必须先完成)

| # | 项目 | 理由 |
|---|------|------|
| B1 | Kahan 公式修复 | P1-02, P1-10 依赖正确的 Kahan 算法 |

P0 修复清单 (来自 CONSOLIDATED_REVIEW / FIX_PLAN):
- B1: Kahan 求和修复 (generic_optim + automagic)
- B2: LLM Adapter in-place → masked_fill
- B3: `'lora'` → `'adapter'` 键名
- B4: cache.py add() 定期 commit
- B5: cache.py weights_only 回退安全

**预计**: 3h

---

### Phase 1: 零风险速修 (独立、无依赖、< 30min/项)

| # | ID | 修复 | 方式 | 预计 |
|---|-----|------|------|------|
| 1 | P1-03 | AdaLN doc-code 不一致 | 战术+战略 | 30min |
| 2 | P1-06 | evaluate 异常 block swap 泄漏 | 战略(context mgr) | 10min |
| 3 | P1-07 | t_embedder LR 归类 | 战术 | 5min |
| 4 | P1-12 | compile_dynamic auto-detect | 战术 | 10min |
| 5 | P1-14 | compile+block_swap 硬互斥 | 战术 | 10min |
| 6 | P1-15 | make_contiguous 冗余 | 战术 | 15min |

**预计**: 1.5h
**验证**: 每项改完即跑 dry-run 10 步确认不崩。

---

### Phase 2: 正确性核心 (依赖 P0 完成)

| # | ID | 修复 | 方式 | 预计 |
|---|-----|------|------|------|
| 7 | P1-02 | Kahan p.grad 复用 | 战术(独立buffer) | 30min |
| 8 | P1-01 | AdaLN 块对角梯度掩码 | 战术 | 15min |
| 9 | P1-16 | reentrant checkpoint 显式 flag | 战术 | 1h |

**预计**: 2h
**验证**: bf16 训练 100 步对比 loss 曲线；开启 activation_checkpointing 跑 swap 场景。

---

### Phase 3: Block swap 重构 (合并处理 P1-08 + P1-09)

| # | ID | 修复 | 方式 | 预计 |
|---|-----|------|------|------|
| 10 | P1-08 | ThreadPool → 主线程同步 | 战术 | 30min |
| 11 | P1-09 | full sync → event sync | 战术 | 45min |

**说明**: P1-08 和 P1-09 紧密耦合（都在 `offloading.py:swap_weight_devices_cuda` 中），建议合并为一个 PR。

**预计**: 1.5h
**验证**: 开启 block_swap 训练 50 步，检查无 illegal loss value；nvidia-smi 监控无 stream 泄漏。

---

### Phase 4: gradient_release 收尾

| # | ID | 修复 | 方式 | 预计 |
|---|-----|------|------|------|
| 12 | P1-04 | gradient_release LR 缩放 | 战术(gas_lr_scale) | 15min |
| 13 | P1-05 | gradient_release add_ 守卫 | 战术(硬守卫) | 15min |
| 14 | P1-13 | per-param→group optimizer | 战术 | 1h |

**预计**: 1.5h
**验证**: 开启 gradient_release 训练 20 步，对比 loss 与标准训练的一致性（允许微小数值差异）。

---

### Phase 5: 性能优化

| # | ID | 修复 | 方式 | 预计 |
|---|-----|------|------|------|
| 15 | P1-11 | attention backend 可配置 | 战术(TE import+配置) | 30min |
| 16 | P1-10 | Kahan PCIe pin_memory + GPU default | 战略+战术 | 30min |

**预计**: 1h
**验证**: P1-11 需确认环境有 TE；P1-10 检查 PCIe 传输时间是否减少。

---

### 暂缓/长期

| # | ID | 说明 |
|---|-----|------|
| — | P1-01 (战略) | AdaLN 三个独立 nn.Linear：需独立分支 + 往返测试 |
| — | P1-04 (战略) | gradient_release 累积梯度模式：需深度重构 |
| — | P1-08+P1-09 (战略) | 真正异步 block swap pipeline：需 CUDA stream 架构重构 |
| — | P1-16 (战略) | checkpoint 集成重构：需与 activation checkpointing 统一 |

---

## 总结

| 阶段 | 项数 | 预计时间 | 累计 |
|------|------|---------|------|
| Phase 0 (P0前置) | 5 | 3h | 3h |
| Phase 1 (零风险速修) | 6 | 1.5h | 4.5h |
| Phase 2 (正确性核心) | 3 | 2h | 6.5h |
| Phase 3 (Block swap) | 2 | 1.5h | 8h |
| Phase 4 (gradient_release) | 3 | 1.5h | 9.5h |
| Phase 5 (性能) | 2 | 1h | 10.5h |
| **合计** | **21** | **10.5h** | |

> **注**: 所有 Phase 1-5 的战术修复总改动量 < 200 行 (不含注释)。战略修复单独评估。
