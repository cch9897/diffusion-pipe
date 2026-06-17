# diffusion-pipe 多Agent交叉审议报告

> 审议日期：2026-06-17 | 3 个独立 Agent 并行交叉审查
> 项目：diffusion-pipe (DeepSpeed pipeline-parallel 扩散模型训练)

---

## Agent 分工

| Agent | 领域 | 审查文件 |
|-------|------|---------|
| A1 | 模型架构 & 正确性 | `models/cosmos_predict2_modeling.py`, `models/cosmos_predict2.py`, `models/base.py`, `models/llm_adapter.py` |
| A2 | 训练基础设施 & 并发 | `train.py`, `optimizers/generic_optim.py`, `optimizers/automagic.py`, `utils/offloading.py`, `utils/unsloth_utils.py`, `utils/patches.py` |
| A3 | 数据管道 & 缓存 | `utils/dataset.py`, `utils/cache.py`, `utils/common.py`, `utils/saver.py`, `utils/isolate_rng.py` |

---

## 交叉投票矩阵

| 问题域 | A1 | A2 | A3 | 共识 |
|--------|:--:|:--:|:--:|:----:|
| Kahan 求和补偿项失效 | — | 🔴 | — | A2 独报（验证确认） |
| automagic 同款 Kahan bug | — | 🔴 | — | A2 独报（验证确认） |
| LLM Adapter in-place 破坏 autograd | 🔴 | — | — | A1 独报（验证确认） |
| AdaLN `adaln_modulation_2` 块对角丢失 | 🔴 | — | — | A1 独报（验证确认，降级 🟡） |
| `train.py:834` `'lora'` → `'adapter'` | — | 🔴 | — | A2 独报（验证确认） |
| `cache.py` weights_only 回退不安全 | — | — | 🔴 | A3 独报（验证确认） |
| `cache.py` `add()` 不 commit | — | — | 🔴 | A3 独报（验证确认） |
| QKV/KV/AdaLN LoRA lora_A 有损平均 | 🟡 | — | — | A1 独报（已知限制，有 warning） |
| CUDA Stream 泄漏 (offloading.py) | — | 🟡 | — | A2 独报 |
| CUDA Event 泄漏 (generic_optim.py) | — | 🟡 | — | A2 独报 |
| `has_inf_or_nan` 溢出风险 | — | 🟡 | — | A2 独报 |
| `_reopen_caches_readonly` 循环引用 | — | — | 🟡 | A3 独报 |
| `mp.Queue()` 跨进程广播风险 | — | — | 🟡 | A3 独报 |
| DataLoader 混合 multiprocess + torch.mp | — | — | 🟡 | A3 独报 |
| 模型卸载内存未验证 | — | — | 🟡 | A3 独报 |
| gradient_release `add_` monkeypatch | — | 🟡 | — | A2 独报 |
| tar 句柄/RoPE dtype 等 | 🟠 | 🟠 | 🟠 | 多 Agent 共报 |

---

## 🔴 阻塞级 (BLOCKING) — 必须立即修复

### B1. Kahan 求和完全失效 (bf16 精度全部丢失)
**位置**: `optimizers/generic_optim.py:489-494`, `optimizers/automagic.py:313-318`
**验证**: ✅ 已确认

```python
# BUG: shift.add_(update) 应为 y = update - shift
# 当前代码使补偿项每步归零，bf16 训练精度持续丢失
shift = state['shift'].to(p.device)
shift.add_(update)          # ← 应为: y = update - shift
p.grad.copy_(p.detach())    # grad = p_old
p.add_(shift)               # p = p_old + (prev_comp + update)
shift.add_(p.grad.sub_(p))  # shift += p_old - (p_old + prev_comp + update) = 0
```

**正确实现**:
```python
y = update - shift                 # y = update - prev_compensation
t = p.detach() + y                 # t = p_old + y
new_shift = (t - p.detach()) - y   # compensation
p.copy_(t)
shift.copy_(new_shift)
```

**影响范围**: 所有使用 bf16 + GenericOptim 或 Automagic 优化器的训练。
**报告 Agent**: A2

### B2. LLM Adapter in-place 操作破坏 autograd
**位置**: `models/cosmos_predict2.py:1023`
**验证**: ✅ 已确认

```python
crossattn_emb[~t5_attn_mask.bool()] = 0  # IN-PLACE on tensor with grad_fn!
```

当 `llm_adapter_lr > 0` 时，`crossattn_emb` 携带 `grad_fn`，此 in-place 赋值导致：
`RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`

**修复**: 替换为 `crossattn_emb = crossattn_emb.masked_fill(~t5_attn_mask.bool().unsqueeze(-1), 0)`
**报告 Agent**: A1

### B3. `communication_data_type` 永远取 model dtype
**位置**: `train.py:834`
**验证**: ✅ 已确认

```python
# BUG: config 中 key 是 'adapter' 而非 'lora'
communication_data_type = config['lora']['dtype'] if 'lora' in config else config['model']['dtype']
```

`'lora' in config` 永远为 `False`，LoRA 训练时 communication_data_type 错误取 model dtype（而非 adapter dtype），导致 DeepSpeed 通信精度不匹配。应为 `'adapter' in config`。
**报告 Agent**: A2

### B4. `cache.py` `add()` 从不 commit SQLite
**位置**: `utils/cache.py:139-163`
**验证**: ✅ 已确认

SQLite 以 `autocommit=False` 打开，`add()` 方法执行 `INSERT` 但不调用 `commit()`。仅 `finalize_current_shard()` (line 113) 和 `init()` (line 85) 提交。长时间缓存生成期间 WAL 日志无限增长，崩溃时数据丢失。应在 `add()` 中每 N 条记录定期提交。
**报告 Agent**: A3

### B5. `cache.py` weights_only 回退不安全
**位置**: `utils/cache.py:40-43`
**验证**: ✅ 已确认

```python
except Exception:
    # Fallback if cached data contains types not supported by weights_only
    item = torch.load(buffer, map_location='cpu')  # 无 weights_only=True
```

宽泛的 `except Exception` 捕获真实错误（数据损坏、磁盘错误），回退到 pickle 反序列化，允许任意代码执行。应替换为特定异常类型 + safetensors 回退路径。
**报告 Agent**: A3

---

## 🟡 高级 (HIGH) — 建议尽快修复

### H1. AdaLN `adaln_modulation_2` 权重块对角结构在训练中被破坏
**位置**: `models/cosmos_predict2_modeling.py:1038-1039` → `models/cosmos_predict2.py:183-185`
**验证**: ✅ 已确认 (原报 🔴，经交叉分析降级为 🟡)

`adaln_modulation_2` 是普通 `nn.Linear`（无结构约束）。初始化为块对角，但训练使非对角块积累非零梯度。保存时 `_split_state_dict_keys` 仅提取对角块，非对角块学习到的信息被静默丢弃。save/load 循环后模型行为不一致。

**影响**: 仅影响 `use_adaln_lora=True` 路径。`use_adaln_lora=False` 路径无此问题（单层 Linear 无块对角结构）。
**修复方案**: 使用 3 个独立 `nn.Linear(R, 3D)` + 在前向中 `torch.cat` 替代单个 `nn.Linear(3R, 9D)`，或实现自定义 Module 强制块对角参数化。
**报告 Agent**: A1

### H2. QKV/KV/AdaLN LoRA `lora_A` 有损平均
**位置**: `models/cosmos_predict2.py:249-255, 279-281, 308-311`
**验证**: ✅ 已确认（已知限制，代码已打印 warning）

旧 LoRA 适配器加载时，多分支 `lora_A` 被平均为单矩阵（3→1 或 2→1）。信息永久丢失。加载后旧适配器行为与原始训练不一致。
**报告 Agent**: A1

### H3. CUDA Stream 泄漏 (`offloading.py`)
**位置**: `utils/offloading.py:71`
**验证**: ✅ 已确认

每次 `swap_weight_devices_cuda` 调用创建新 `torch.cuda.Stream()`，未复用。长时间训练累积 CUDA 资源。应创建持久 stream 复用。
**报告 Agent**: A2

### H4. CUDA Event 泄漏 (`generic_optim.py`)
**位置**: `optimizers/generic_optim.py:504-505`
**验证**: ✅ 已确认

每个 optimizer step 创建新 `torch.cuda.Event()`。应复用同一个 event。
**报告 Agent**: A2

### H5. `has_inf_or_nan` fp16/bf16 溢出风险
**位置**: `optimizers/generic_optim.py:23-25`
**验证**: ✅ 已确认

`x.sum()` 在 fp16（范围 ±65504）或 bf16 大张量上可能溢出，误报 Inf。应改用 `.float().sum()` 或逐元素检查。
**报告 Agent**: A2

### H6. `_reopen_caches_readonly` 循环引用风险
**位置**: `utils/dataset.py:1309-1319`
**验证**: ✅ 已确认

递归遍历 `__dict__` 无 visited 集合，循环引用导致无限递归 RecursionError。
**报告 Agent**: A3

### H7. `mp.Queue()` 跨进程广播风险
**位置**: `utils/dataset.py:1162-1167`
**验证**: ✅ 已确认

`broadcast_object_list` 广播 `mp.Manager().Queue()` 对象本身，非主进程接收代理引用。DeepSpeed 进程组初始化不同步时可能失败。
**报告 Agent**: A3

### H8. 模型卸载内存未验证
**位置**: `utils/dataset.py:1196-1207`
**验证**: ✅ 已确认（代码含 `TODO: check if this is actually freeing memory`）

如果内存实际未释放，多数据集处理会累积 GPU 内存泄漏。
**报告 Agent**: A3

### H9. gradient_release `add_` 全局 monkeypatch
**位置**: `train.py:712-715`
**验证**: ✅ 已确认

注释自认 "not mathematically sound"。替换所有可训练参数的 `add_` 为 `data.add_`，绕过 autograd in-place 检测。仅 gradient_release 模式触发，但一旦激活全局生效。非梯度释放路径的 `p.add_()` 调用会静默破坏计算图。
**报告 Agent**: A2

---

## 🟠 中级 (MEDIUM)

| # | 描述 | 位置 |
|---|------|------|
| M1 | `tarfile_map` 字典累积打开 tar 句柄永不关闭 → 文件描述符耗尽 | `dataset.py:771-772` |
| M2 | `pool.close()` 缺少 `pool.join()` → 子进程僵尸 | `dataset.py:158` |
| M3 | `cache.py` `open_files` 字典无限增长文件句柄（>1000 分片时描述符耗尽） | `cache.py:30` |
| M4 | `cache.py` `items` 表缺少主键 → 可能重复插入 | `cache.py:66` |
| M5 | `cache.py` `init_readonly()` 多 worker 共享 `self` → close/open 竞争 | `cache.py:132` |
| M6 | `cache.py` `clear()` 无 try/except → 其他进程使用文件时崩溃 | `cache.py:88-94` |
| M7 | `DotProductAttention` 无 import（TE 后端不可达） | `modeling.py:424` |
| M8 | unsloth `reentrant_activation_checkpointing` 强制覆盖为 True | `train.py:103-104` |
| M9 | `unsloth_utils.py` pinned/non-pinned D2H 路径不一致 | `unsloth_utils.py:33-37` |
| M10 | offloading.py `torch.is_grad_enabled()` 检测 reentrant checkpoint 不够精确 | `offloading.py:287-289` |

---

## 🟢 低级 (LOW)

| # | 描述 | 位置 |
|---|------|------|
| L1 | `KEEP_IN_HIGH_PRECISION` 模糊匹配 `final_layer` / `x_embedder` 可能覆盖过多子模块 | `cosmos_predict2.py:33,684` |
| L2 | `llm_adapter.py` 独立定义 RMSNorm/RoPE，与主模型不一致 | `llm_adapter.py:18-48` |
| L3 | `fused=True` RoPE 路径永久不可达 (`assert fused == False`) | `modeling.py:211` |
| L4 | `DTYPE_MAP` 缺少常用别名 (`'fp16'`, `'fp32'`) | `common.py:14-21` |
| L5 | `patches.py` `torch._six` 回退依赖已废弃 API | `patches.py:20-23` |
| L6 | Ruff lint: F401, E712, F841, E702 (×6), E741 (×2), F402, E711 | 多文件 |

---

## 热点文件矩阵

多 Agent 共报 = 高优先级关注：

| 文件 | A1 | A2 | A3 | 热点等级 |
|------|:--:|:--:|:--:|:--------:|
| `optimizers/generic_optim.py` | — | 🔴 B1 + 🟡 H4,H5 | — | ⭐⭐⭐ |
| `utils/cache.py` | — | — | 🔴 B4,B5 + 🟠 M4-6 | ⭐⭐⭐ |
| `models/cosmos_predict2_modeling.py` | 🔴 B2 + 🟡 H1 | — | — | ⭐⭐⭐ |
| `models/cosmos_predict2.py` | 🔴 B2 + 🟡 H1,H2 | — | — | ⭐⭐⭐ |
| `train.py` | — | 🔴 B3 + 🟡 H9 | — | ⭐⭐ |
| `utils/offloading.py` | — | 🟡 H3 + 🟠 M10 | — | ⭐⭐ |
| `utils/dataset.py` | — | — | 🟡 H7, H8 + 🟠 M1, M2 | ⭐⭐ |
| `utils/unsloth_utils.py` | — | 🟠 M9 | — | ⭐ |
| `optimizers/automagic.py` | — | 🔴 B1 | — | ⭐⭐ |

---

## 修复路线

### 阶段 1: 立即修复 (P0 — 阻塞正确性)

| 优先级 | 修复项 | 工作量 | 影响 |
|--------|--------|--------|------|
| 🔴 P0 | B1: Kahan 求和修复 (generic_optim + automagic) | 1h | 修复所有 bf16 训练精度 |
| 🔴 P0 | B2: LLM Adapter in-place → masked_fill | 5min | 修复 llm_adapter 可训练崩溃 |
| 🔴 P0 | B3: `'lora'` → `'adapter'` 键名 | 1min | 修复 LoRA 通信精度 |
| 🔴 P0 | B4: cache.py add() 定期 commit | 30min | 防止缓存数据丢失 |
| 🔴 P0 | B5: cache.py weights_only 回退安全 | 30min | 修复安全风险 |

### 阶段 2: 短期修复 (P1 — 高优先级)

| 优先级 | 修复项 | 工作量 |
|--------|--------|--------|
| 🟡 P1 | H1: AdaLN mm.nn.Module 强制块对角 | 4h (3 个独立 nn.Linear 替代) |
| 🟡 P1 | H3+H4: CUDA Stream/Event 复用 | 1h |
| 🟡 P1 | H5: has_inf_or_nan float sum | 5min |
| 🟡 P1 | H6: _reopen_caches_readonly visited set | 10min |
| 🟡 P1 | H7: mp.Queue 独立进程创建 | 2h |
| 🟡 P1 | H9: gradient_release add_ 文档 + 隔离 | 1h |

### 阶段 3: 后续改进 (P2 — 稳健性)

| 优先级 | 修复项 | 工作量 |
|--------|--------|--------|
| 🟠 P2 | M1-6: cache/dataset 资源泄漏修复 | 3h |
| 🟠 P2 | M9: unsloth D2H 路径统一 | 30min |
| 🟠 P2 | H2: lora_A 有损平均（已知限制，暂维持现状但改善 warning） | 1h |

---

## Ruff Lint 汇总

| 文件 | 问题 |
|------|------|
| `models/cosmos_predict2_modeling.py` | F401 (unused import), E712 (assert not), F841 (unused var), F821 (undefined name), E702 (×6, semicolons) |
| `utils/dataset.py` | E741 (×2, ambiguous var), F402 (shadowed import), E711 (is None) |

---

*审议完成时间: 2026-06-17*
*各 Agent 独立报告见下属会话*
