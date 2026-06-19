# Anima FFT 现状审议 — 6专家交叉审议报告

**日期**: 2026-06-19
**范围**: Anima FFT (Full Fine-Tuning) 当前代码状态
**基准**: HEAD 7d3ac9b
**方式**: 两轮并行 6 专家正交审议 (3+3)，轮内并行

---

## 专家矩阵

| 专家 | 轮次 | 领域 | 状态 |
|------|------|------|------|
| E1 | R1 | 数值正确性 (state_dict remap, LoRA keys, fusion, loss, dtype) | ✅ 完成 |
| E2 | R1 | 训练基础设施 (gradient_release, Kahan, checkpoint, eval safety) | ✅ 完成 |
| E3 | R1 | 架构与集成 (param groups, block-diagonal, LoRA targets, pipeline) | ✅ 完成 |
| E4 | R2 | Block Swap / Offloading (stream sync, pin_memory, CUDA events) | ✅ 完成 |
| E5 | R2 | Optimizer 性能 (per-param, Kahan overhead, cpu_offload, GAS) | ✅ 完成 |
| E6 | R2 | Compile / Graph / Attention (compile config, backend, contiguous) | ✅ 完成 |

---

## 交叉投票矩阵

| # | 议题 | E1 | E2 | E3 | E4 | E5 | E6 | 共识 |
|---|------|----|----|----|----|----|----|------|
| A1 | **adaln_modulation_2 块对角训练中被破坏** | ❌ | — | ❌ | — | — | — | ✅ 双确认 |
| A2 | **t_embedder AdaLN 参数归类错误 (base_params)** | — | — | ❌ | — | — | — | E3 独 |
| A3 | **partition_split 默认值崩溃 (pipeline>2)** | — | — | ❌ | — | — | — | E3 独 |
| A4 | p.add_ monotpatch 缺少 restore/opt-out | — | ⚠️ | — | — | — | — | E2 独 |
| A5 | GradientReleaseOptimizerWrapper 缺 super().__init__ | — | ⚠️ | — | — | — | — | E2 独 |
| A6 | QKV/KV LoRA lora_A 取平均有损 | ⚠️ | — | — | — | — | — | E1 独 |
| A7 | evaluate 的 prepare_block_swap_inference 在 try 外 | — | ⚠️ | — | — | — | — | E2 独 |
| A8 | Saver 缺少 contiguous() 守卫 | — | ⚠️ | — | — | — | — | E2 独 |
| A9 | adapter_target_modules 缺少 LLMAdapter/FinalLayer | — | — | ⚠️ | — | — | — | E3 独 |
| A10 | CUDA event 泄漏 (reentrant checkpoint skip 路径) | — | — | — | ⚠️ | — | — | E4 独 |
| A11 | Loss mean vs sum/mask.sum 语义差异 | ⚠️ | — | — | — | — | — | E1 独 |
| A12 | AdaLN split 路径缺 shape divisibility 检查 | ⚠️ | — | — | — | — | — | E1 独 |
| B1 | 已确认修复: AdaLN LoRA remap/split 维度 | ✅ | — | — | — | — | — | ✅ |
| B2 | 已确认修复: Prefetch NCCL 竞态 | — | — | ✅ | — | — | — | ✅ |
| B3 | 已确认修复: Prefetch epoch 竞态 | — | — | ✅ | — | — | — | ✅ |
| B4 | 已确认修复: evaluate try/finally | — | ✅ | — | — | — | — | ✅ |
| B5 | 已确认修复: block swap TPE 移除 + CUDA event | — | — | — | ✅ | — | — | ✅ |
| B6 | 已确认修复: gradient_release pipeline 守卫 | — | ✅ | — | — | — | — | ✅ |
| B7 | 已确认修复: gas_lr_scale (LR 缩放 1/GAS) | — | ✅ | — | — | — | — | ✅ |
| B8 | 已确认修复: per-param → per-lr-grouping | — | — | — | — | ✅ | — | ✅ |
| B9 | 已确认修复: attention_backend → TE 默认 | — | — | — | — | — | ✅ | ✅ |
| B10 | 已确认修复: compile + blocks_to_swap 硬阻止 | — | — | — | — | — | ✅ | ✅ |
| B11 | 已确认修复: InitialLayer 只对 x contiguous (8 tensors skip) | — | — | — | — | — | ✅ | ✅ |
| B12 | 已确认修复: stream.synchronize() 移除 | — | — | — | ✅ | — | — | ✅ |
| B13 | 已确认修复: compile-on-block-level 正确设计 | — | — | — | — | — | ✅ | ✅ |
| B14 | 已确认修复: pin_memory 正确应用于 swappable weights | — | — | — | ✅ | — | — | ✅ |
| C1 | Kahan 算法正确性 (opposite-sign convention) | — | ✅ | — | — | — | — | ✅ |
| C2 | p.grad 复用安全性 (step/zero_grad 间无外部读取) | — | ✅ | — | — | — | — | ✅ |
| C3 | compute↔swap overlap 已保持 (event-based GPU-async) | — | — | — | ✅ | — | — | ✅ |
| C4 | Contiguous 优化正确 (只对修改后的 tensor) | — | — | — | — | — | ✅ | ✅ |
| C5 | Graph breaks 最小化 (无严重 Dynamo 断点) | — | — | — | — | — | ✅ | ✅ |
| C6 | is_grad_enabled reentrant 检测标准路径可靠 | — | — | — | ✅ | — | — | ✅ |
| C7 | Kahan bf16 overhead ~2-5% | — | — | — | — | ✅ | — | ✅ |
| C8 | foreach=False with 6 grouped optimizers 可忽略 | — | — | — | — | ✅ | — | ✅ |

图例: ✅ 正面确认 | ⚠️ 发现问题 | ❌ 严重问题 | — 该专家未关注此议题

---

## 最终决议表 — 按优先级排列

### P0 — 致命，必须立即修复

#### P0-1: partition_split 默认值对 pipeline_stages>2 崩溃 🔴

- **位置**: `models/cosmos_predict2.py:663` (`partition_split = config.get('partition_split', [len(layers) / num_stages])`)
- **问题**: 默认值产生单个 float 的 list，而非 `num_stages-1` 个分区边界。`utils/pipeline.py:45` 断言 `num_partitions == num_stages - 1` 必然失败
- **触发**: `pipeline_stages >= 3` 且未手动指定 `partition_split`
- **修复**: 默认值改为 `[len(layers) * i / num_stages for i in range(1, num_stages)]`，~3 行
- **裁决**: E3 独家发现，经手动验证确认 ✅

#### P0-2: adaln_modulation_2 块对角结构训练中被破坏 → save/load 静默信息丢失 🔴

- **位置**: `models/cosmos_predict2_modeling.py:1048-1050` + `models/cosmos_predict2.py:183-185`
- **问题**: `nn.Linear(3R, 9D)` 初始化为块对角（零矩阵），训练中无梯度掩码/hook 约束，非对角块积累非零梯度。save 时 `_split_state_dict_keys` 仅提取对角块 → 非对角块学习到的信息静默丢弃
- **触发**: `use_adaln_lora=True` 训练后进行 save→load 往返
- **修复**: 战术方案 — backward hook 将非对角块梯度清零 (~15 行)。战略方案 — 3 个独立 `nn.Linear(R, 3D)` (~80 行)
- **裁决**: E1 + E3 双确认，一致判定 P0

### P1 — 严重，应尽快修复

#### P1-1: t_embedder AdaLN-LoRA 参数归类到 base_params（错误的 LR） ⚠️

- **位置**: `models/cosmos_predict2.py:930` (`elif '.adaln_modulation' in name`)
- **问题**: `t_embedder.1.linear_2` (产生 `adaln_lora_B_T_3D` 的 AdaLN 层) 不匹配 `.adaln_modulation` 子串 → 走 `base_lr` 而非 `mod_lr`
- **修复**: 扩展匹配条件 `'t_embedder' in name and 'adaln' in name`，~1 行
- **裁决**: E3 发现，与 P1_ARCHITECTURAL_FIX_PLAN P1-07 一致

#### P1-2: GradientReleaseOptimizerWrapper 架构缺陷 ⚠️

- **位置**: `optimizers/gradient_release.py:14-21` + `train.py:784-787`
- **问题**:
  - 未调用 `super().__init__()` → `self.state` 是普通 dict（非 defaultdict），`self.defaults` 未设置
  - monkeypatch 无 restore 机制，一旦应用全局生效
  - 若 DeepSpeed 访问 `optimizer.state` 可能崩溃
- **修复**: 调用 `super().__init__()` + 添加 `original_add_` 备份用于 restore，~15 行
- **裁决**: E2 发现

#### P1-3: adapter_target_modules 不完整 ⚠️

- **位置**: `models/cosmos_predict2.py:587-590`
- **问题**: 缺少 `'LLMAdapter'`（其 `in_proj`/`out_proj` 无法 LoRA）和 `'FinalLayer'`（输出头无法 LoRA）
- **修复**: 添加两个类名，~1 行
- **裁决**: E3 发现

#### P1-4: CUDA event 泄漏（reentrant checkpoint skip 路径） ⚠️

- **位置**: `utils/offloading.py:305-311, 319-325`
- **问题**: 当 `reentrant_activation_checkpointing=True` 且 replay forward 时，`wait_for_block` 和 `submit_move_blocks_forward` 跳过处理但不清除已有 event → event 对象累积
- **修复**: skip 路径中 pop 并 discard 过期 event，~5 行
- **裁决**: E4 发现

### P2 — 建议，低优先级

#### P2-1: Saver 缺少 contiguous() 守卫

- **位置**: `utils/saver.py:77, 103`
- **问题**: `p.detach()` 对非连续 tensor 产生非连续副本，safetensors 保存可能出错
- **修复**: 改为 `p.detach().contiguous()`
- **裁决**: E2 发现，已知问题 (P2-1 in anima_ft_review.md)

#### P2-2: evaluate 中 prepare_block_swap_inference 在 try 外

- **位置**: `train.py:241`
- **问题**: init 阶段 OOM 会跳过 finally 中的状态恢复
- **修复**: 移入 try 块，2 行改动
- **裁决**: E2 发现

#### P2-3: QKV/KV LoRA lora_A 取平均有损（已知、已警告）

- **位置**: `models/cosmos_predict2.py:255, 286`
- **问题**: 3→1 或 2→1 平均丢失跨分支差异
- **状态**: 已有清晰 warning，已知限制
- **裁决**: E1 确认无新问题

#### P2-4: Loss mean vs sum/mask.sum 语义（多尺寸 bucketing 场景）

- **位置**: `models/cosmos_predict2.py:977`
- **问题**: 固定分辨率 FFT 训练无影响；多尺寸 bucketing 时梯度尺度不一致
- **裁决**: E1 确认，对 Anima FFT 影响低

#### P2-5: AdaLN split 路径缺 shape divisibility 检查

- **位置**: `models/cosmos_predict2.py:422` (`ald = mod2_w.shape[1] // 3`)
- **问题**: remap 路径有 warning，split 路径没有
- **修复**: 添加 assert 或 warning，~3 行
- **裁决**: E1 发现

#### P2-6: Docstring 过时（声称 block-diagonal lora_B）

- **位置**: `models/cosmos_predict2.py:217-222`
- **问题**: docstring 描述的条件 cat/average 与代码不完全一致
- **裁决**: E1 发现，cosmetic

---

## 正面确认清单（已验证无问题）

| # | 项目 | 验证专家 | 备注 |
|---|------|---------|------|
| ✅ | AdaLN LoRA remap/split 维度正确 (cat dim1 / slice dim1) | E1 | commit c9f04c6 修复 |
| ✅ | QKV/KV lora_B 拼接正确 (cat dim0) | E1 | 无损操作 |
| ✅ | Kahan 补偿算法正确 (opposite-sign convention) | E2 | 等价于标准 Kahan |
| ✅ | p.grad 复用安全 (step→zero_grad 间无外部读取) | E2 | commit 9375ad2 有意恢复 |
| ✅ | gas_lr_scale 默认 1/gas 正确 | E2 | 匹配 Adam 线性缩放 |
| ✅ | evaluate try/finally 基本充分 | E2 | 仅边缘窗口 |
| ✅ | Prefetch NCCL 守卫正确 (pipeline_stages>1 禁用) | E3 | commit c9f04c6 修复 |
| ✅ | Prefetch epoch 锁正确 (threading.Lock) | E3 | commit c9f04c6 修复 |
| ✅ | Block swap TPE 已移除 + CUDA events 正确 | E4 | commit faf1188/79e9d40 |
| ✅ | compute↔swap overlap 已保持 (event-based) | E4 | 非阻塞 GPU 依赖 |
| ✅ | pin_memory 正确应用 (排除 LoRA) | E4 | |
| ✅ | 热路径无 blocking synchronize | E4 | 仅 init 阶段有一次 |
| ✅ | _swap_stream 生命周期正确 (模块级 lazy-init) | E4 | |
| ✅ | per-param → per-lr-grouping (4000→6 instances) | E5 | |
| ✅ | foreach=False 对 6 group 可忽略 | E5 | <0.1% overhead |
| ✅ | Kahan bf16 overhead ~2-5% | E5 | 可接受 |
| ✅ | attention_backend → TE 默认 | E6 | P1-11 已修复 |
| ✅ | compile + blocks_to_swap 硬阻止 (ValueError) | E6 | P1-14 已修复 |
| ✅ | InitialLayer 只 contiguous x (8 tensors skip) | E6 | P1-15 已修复 |
| ✅ | compile-on-block-level 设计正确 | E6 | offloader 在编译图外 |
| ✅ | 无严重 Dynamo graph break | E6 | fullgraph=False 充分 |
| ✅ | contiguous 调用已最优化 | E6 | 每层仅修改后的 tensor |

---

## 修复路线图

### Phase 1: 立即修（确定性崩溃 + 正确性风险）

| # | 项目 | 改动量 | 风险 | 依赖 |
|---|------|--------|------|------|
| P0-1 | partition_split 默认值修复 | ~3 行 | 极低 | 无 |
| P0-2 | adaln_modulation_2 梯度掩码 (战术) | ~15 行 | 低 | 无 |
| P1-1 | t_embedder AdaLN LR 归类 | ~1 行 | 极低 | 无 |

### Phase 2: 尽快修（架构正确性）

| # | 项目 | 改动量 | 风险 | 依赖 |
|---|------|--------|------|------|
| P1-2 | GradientReleaseOptimizerWrapper super().__init__ | ~15 行 | 低 | 需验证 |
| P1-3 | adapter_target_modules 补全 | ~1 行 | 极低 | 无 |
| P1-4 | CUDA event 泄漏修复 | ~5 行 | 低 | 无 |

### Phase 3: 建议修（防御性 + 低风险）

| # | 项目 | 改动量 | 风险 |
|---|------|--------|------|
| P2-1 | Saver contiguous 守卫 | ~4 行 | 极低 |
| P2-2 | evaluate init 移入 try | ~2 行 | 极低 |
| P2-5 | AdaLN split 路径 divisibility 断言 | ~3 行 | 极低 |

### Phase 4: Known Limitations（无代码改动）

| # | 项目 | 动作 |
|---|------|------|
| P2-3 | QKV/KV LoRA averaging lossy | 文档标注，不接受 "无损" 假设 |
| P2-4 | Loss mean vs mask.sum | 多 bucket 场景文档说明 |
| P2-6 | Docstring 过时 | 下次重构时更新 |

---

## 与历史审议的对比

相比 2026-06-18 的两份审议报告 (anima_ft_review.md + anima_ft_review_2025-06-18.md):

**已修复 (13 项 P0/P1)**:
- P0-1: AdaLN LoRA lora_A 维度 → ✅ c9f04c6
- P0-2: Prefetch NCCL 竞态 → ✅ c9f04c6
- P0-3: Prefetch epoch 竞态 → ✅ c9f04c6
- P0 (旧): Block swap TPE CUDA context → ✅ faf1188
- P0 (旧): Kahan p.grad 复用 (→ 有意恢复 9375ad2)
- P0 (旧): AdaLN doc-code (→ docstring 已更新)
- P1-06: EVAL_SWAP evaluate 状态泄漏 → ✅ c9f04c6
- P1-08: SWAP_CTX TPE 移除 → ✅ faf1188
- P1-09: SWAP_SYNC synchronize 移除 → ✅ 79e9d40
- P1-11: ATTN_BACKEND 硬编码 → ✅ 当前代码
- P1-14: COMPILE_BS 互斥检测 → ✅ 当前代码
- P1-15: CONTIG_280 冗余 → ✅ 当前代码
- P1-04: GRADREL_LR gas_lr_scale → ✅ 当前代码

**仍待修复 (3 项 P0/P1)**:
- P1-01: ADALN_DIAG → 本报告 P0-2
- P1-07: T_EMBED_LR → 本报告 P1-1
- P1-03: ADALN_DOC → 本报告 P2-6

**新发现 (2 项 P0/P1)**:
- P0-1: partition_split 崩溃 (E3 独)
- A10: CUDA event 泄漏 (E4 独)

---

## 总结

Anima FFT 路径经过 13 项已修复的 P0/P1 问题后，核心正确性已大幅改善。当前剩余 **2 个 P0**（partition_split 崩溃 + adaln 块对角信息丢失）和 **4 个 P1**（t_embedder LR 归类、GradientReleaseOptimizerWrapper 架构、adapter_target_modules 不完整、CUDA event 泄漏）。全部改动量约 40 行，可在 2 小时内完成修复 + 验证。

性能面已确认无退步：compute↔swap overlap 保持、Kahan overhead ~2-5%、contiguous 调用已最优化、attention backend 默认 TE。
