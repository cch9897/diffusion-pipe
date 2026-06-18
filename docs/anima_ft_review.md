# Anima FT 多 Agent 交叉审议报告

**日期**: 2026-06-18
**审议范围**: Anima FT 相关代码 — `models/cosmos_predict2.py` (1080行) + `models/cosmos_predict2_modeling.py` (1485行) + `train.py` (994行) + `models/base.py` + `optimizers/`
**审议方式**: 三个正交领域 Agent 并行审议 + 关键发现独立验证

- **Agent A** — 数值正确性: QKV/AdaLN 融合、state_dict 重映射、LoRA key 处理、dtype 一致性
- **Agent B** — 性能调优: torch.compile、显存/block swap、gradient_release、Kahan 效率、attention 瓶颈
- **Agent C** — 并发安全与严重 Bug: DataLoader 线程、设备处理、边界条件、错误路径、序列化竞态

---

## 交叉投票矩阵

| 议题 | Agent A | Agent B | Agent C | 共识 |
|------|---------|---------|---------|------|
| QKV 融合数值正确性 | ✅ | ✅ | — | ✅ 通过 |
| AdaLN 融合顺序正确性(非LoRA) | ✅ | ✅ | — | ✅ 通过 |
| state_dict 权重重映射 | ✅ | ✅ | — | ✅ 通过 |
| LoRA key 转换(AdaLN lora_A) | ❌ | — | — | ❌ P0 Bug |
| RoPE dtype 处理 | ✅ | ✅ | — | ✅ 通过 |
| 学习率分组(融合后) | ✅ | — | — | ✅ 通过 |
| torch.compile 配置 | — | ⚠️ | — | ⚠️ 可改进 |
| Kahan 求和实现 | — | ⚠️ | ✅ | ⚠️ PCIe 开销 |
| gradient_release 正确性 | — | ❌ | ⚠️ | ❌ LR 未缩放 |
| block swap 效率/状态管理 | — | ⚠️ | ❌ | ❌ eval 异常泄漏 |
| attention 实现 | — | ❌ | — | ❌ backend 硬编码 |
| 融合操作零拷贝性 | — | ✅ | — | ✅ 最优 |
| 显存管理 | — | ⚠️ | — | ⚠️ unsloth 冲突 |
| DataLoader fork 安全 | — | — | ✅ | ✅ 已修复 |
| 序列化 contiguous guard | — | — | ⚠️ | ⚠️ base.py 遗漏 |
| original_name getattr 守卫 | — | — | ⚠️ | ⚠️ sdxl.py 遗漏 |
| 设备处理 | — | — | ✅ | ✅ 安全 |
| 错误路径处理 | — | — | ⚠️ | ⚠️ resume 脆弱 |
| 数值边界 | — | — | ✅ | ✅ 安全 |
| Prefetch 线程竞态 | — | — | ❌ | ❌ NCCL/epoch 竞态 |

---

## 最终决议表 — 按优先级排列

### P0 — 致命，必须立即修复

#### P0-1: AdaLN LoRA `adaln_modulation_2.lora_A` 维度不匹配 [已验证]

- **位置**: `models/cosmos_predict2.py:320-321` (remap), `:415` (split)
- **问题**: `adapter_target_modules` 含 `'Block'`（line 577-578），PEFT 为 `adaln_modulation_2 = Linear(3*ald, 9*D)` 创建 LoRA
  - **remap** (`_remap_lora_keys`): `sum(l2_list)/3` 产生 `(R, ald)`，但 fused Linear 需要 `lora_A = (R, 3*ald)` → 加载旧格式 LoRA 时维度不匹配崩溃
  - **split** (`_split_lora_keys`): `mod2_w.clone()` 产生 `(R, 3*ald)`，但 legacy Linear 需要 `lora_A = (R, ald)` → 保存的 adapter 无法被 ComfyUI 加载
- **验证**: 用 Python 脚本确认维度：remap 产生 (16,256)，fused 需要 (16,768)；split 产生 (16,768)，legacy 需要 (16,256)
- **修复**:
  - remap 改为 `torch.cat(l2_list, dim=1)` → `(R, 3*ald)`
  - split 改为 `mod2_w[:, i*ald:(i+1)*ald]` → `(R, ald)`
- **触发条件**: 加载旧格式(3组分离)的 AdaLN LoRA adapter 时；新训练不受影响
- **严重性**: 确定性崩溃，非偶发

#### P0-2: Prefetch 线程与主线程 NCCL 通信竞态 [Agent C]

- **位置**: `train.py:934-935` + `utils/dataset.py:1436-1441`
- **问题**: `pipeline_stages > 1` 时，prefetch 线程在后台执行 `dist.send`/`dist.recv`（使用默认 process group），主线程同时执行 `model_engine.train_batch` 的 NCCL 通信。NCCL 不保证同一 communicator 上并发操作线程安全 → 死锁或静默数据损坏
- **修复**: prefetch 线程中的集合通信使用独立 process group，或改为同步预取
- **触发条件**: 多卡流水线并行 (`pipeline_stages > 1`) + prefetch

#### P0-3: Prefetch 线程与 sync_epoch 数据竞态 [Agent C]

- **位置**: `train.py:934,938` + `utils/dataset.py:1392`
- **问题**: prefetch 线程在 `__next__` 的 `StopIteration` 分支中递增 `self.epoch`，主线程 `sync_epoch()` 读取 + `all_gather_object` → epoch 不同步
  - 导致: 学习率调度提前推进、`saver.process_epoch` 误判 `finished_epoch`、epoch loss 统计错误
- **修复**: epoch 递增移到主线程，或加锁
- **触发条件**: 启用 prefetch (first/last stage)

---

### P1 — 严重，应尽快修复

#### P1-1: evaluate 异常时 block swap 状态泄漏 [Agent C]

- **位置**: `train.py:237-249`
- **问题**: `prepare_block_swap_inference()` (设置 `forward_only=True`) 和 `prepare_block_swap_training()` 不在 try/finally 中。评估异常后模型永久卡在 `forward_only=True` → backward hook 跳过 block swap → 被换出 block 的梯度丢失 → 训练静默损坏
- **修复**: 将 evaluate 调用包入 `try:`，`finally:` 中恢复 training 状态
- **改动量**: ~5 行

#### P1-2: attention backend 硬编码为 'torch' [已验证]

- **位置**: `models/cosmos_predict2_modeling.py:1215`
- **问题**: `atten_backend = 'torch'` 硬编码，覆盖 `Attention` 类默认的 `'transformer_engine'`。放弃 TE DotProductAttention (FlashAttention-2 fused kernel)，长序列 attention 慢 2-3x
- **验证**: 确认 line 1215 `atten_backend = 'torch'`，传入 `self_attention_backend`/`cross_attention_backend`（line 1274-1275），而 `Attention.__init__` 默认 `backend='transformer_engine'`（line 394）
- **修复**: 改为可配置 `self.model_config.get('attention_backend', 'torch')`，或在 TE 可用时自动启用
- **预估收益**: 15-30% 端到端加速 (attention 占 transformer FLOPs 40-50%)
- **前提**: 需确认训练机是否安装 transformer_engine

#### P1-3: gradient_release 学习率未按 GAS 缩放 [Agent B+C 共识]

- **位置**: `train.py:746-748`
- **问题**: betas 按 `**(1/gas)` 缩放补偿更频繁更新，但 LR 未缩放。每 micro-batch 执行完整 `optimizer.step()`，等效更新频率提高 GAS 倍。代码注释自承 "unbelievably hacky and not mathematically sound"
- **争议**: momentum scaling 使每步 update 变小，是否完全抵消取决于 Adam 动力学
- **修复**: LR 应除以 GAS，或至少提供 `gas_lr_scale` 配置项

#### P1-4: torch_compile_dynamic 默认 False [Agent B]

- **位置**: `models/cosmos_predict2.py:836`
- **问题**: 多 bucket 训练时每个 (block, shape) 组合触发独立 recompile。28 blocks × N buckets = 28N 次编译，每次数十秒 + 显存膨胀
- **修复**: 多 bucket 场景默认 True，或文档明确提示
- **预估收益**: 编译时间 -80%

#### P1-5: Kahan CPU offload PCIe 双向同步 [Agent B]

- **位置**: `optimizers/generic_optim.py:492,498`
- **问题**: `kahan_buffer_device='cpu'` 时每步全参数 PCIe round-trip。H2D 用 `non_blocking=True` 但 CPU 内存未 pin → 退化为同步。7B 模型约 14GB/step PCIe 传输，PCIe Gen4 x16 下约 0.44s/step
- **修复**: 使用 `pin_memory()`，或评估是否真需要 CPU offload kahan buffer

#### P1-6: block swap 内 full synchronize 阻塞 pipeline [Agent B]

- **位置**: `utils/offloading.py:74`
- **问题**: 每次权重交换前 `torch.cuda.current_stream().synchronize()` 全同步，stall 主计算流。增加 1-3ms/swap × 2×blocks_to_swap/step
- **修复**: 使用独立 CUDA stream + event 同步替代全同步

#### P1-7: per-param optimizer + foreach=False 大量小 kernel [Agent B]

- **位置**: `train.py:721-722,741,744`
- **问题**: 每参数独立 optimizer 实例，`foreach=False`，数千小 kernel launch/step。对比 foreach=True 批量 kernel，kernel launch overhead 增加 5-10x
- **修复**: 按 param group 合并 optimizer，或用 foreach=True 的批量 kernel

#### P1-8: blocks_to_swap 与 torch.compile recompile 冲突 [Agent B]

- **位置**: `models/cosmos_predict2.py:841-842`
- **问题**: block swap 通过 `weight.data` 原地修改触发 dynamo guard 失效 → recompile。代码已有 warning 但未阻止
- **修复**: 二者互斥检测，或 `dynamic=True` 缓解

---

### P2 — 建议，低优先级

#### P2-1: base.py save_adapter/save_model 缺少 contiguous() 守卫 [Agent C]

- **位置**: `models/base.py:605-609,635-636`
- cosmos_predict2.py 已修复 (commit 5bc090b)，base.py 遗漏。防御性补齐

#### P2-2: sdxl.py original_name 缺少 getattr 守卫 [Agent C]

- **位置**: `models/sdxl.py:605-612`
- commit 35371a5 修复了其他文件，sdxl.py 遗漏。仅影响 SDXL，不影响 Anima

#### P2-3: checkpoint resume 错误处理不完善 [Agent C]

- **位置**: `train.py:863-884`
- 硬 assert + 无 key 检查 + 无 try/except。checkpoint 损坏时崩溃信息不直观

#### P2-4: convert_state_dict_dtype 在循环内调用 (O(n²)) [Agent C]

- **位置**: `utils/saver.py:74-76`
- 每次迭代转换整个 dict，应移到循环外

#### P2-5: torch_attention_op 输出 reshape 隐式 copy [Agent B]

- **位置**: `models/cosmos_predict2_modeling.py:346-349`
- SDPA 输出 transpose 后 reshape 触发 copy，56 次/step

#### P2-6: LoRA remap/split 缺少测试覆盖 [Agent A]

- **位置**: `tests/test_state_dict_remap.py`
- 仅覆盖权重 remap/split，未覆盖 LoRA。P0-1 不会被现有测试发现

#### P2-7: docstring 与实现不一致 [Agent A]

- **位置**: `models/cosmos_predict2.py:217-222`
- docstring 描述 "条件 cat/average" 但代码无条件 average

#### P2-8: disable_block_swap 重复调用丢失原始值 [Agent C]

- **位置**: `utils/offloading.py:223-229`
- 非配对调用时 `blocks_to_swap_tmp` 被覆写为 None。正常流程不触发

#### P2-9: load_diffusion_model 未验证 state_dict 完整性 [Agent C]

- **位置**: `models/cosmos_predict2.py:691-695`
- 缺失参数跳过不报错，meta device 参数延迟崩溃

---

## 修复建议优先级排序

### 立即修 (确定性崩溃)

1. **P0-1** AdaLN LoRA lora_A 维度 — 2 行改动，cat dim1 / slice dim1
2. **P1-1** evaluate try/finally — 5 行改动，防梯度丢失

### 多卡流水线并行用户必修

3. **P0-2** Prefetch NCCL 竞态
4. **P0-3** Prefetch epoch 竞态

### 性能优化 (按收益排序)

5. **P1-2** attention backend 改回 TE — 预估 15-30% 加速
6. **P1-4** dynamic=True 默认 — 多 bucket 编译时间 -80%
7. **P1-3** gradient_release LR 缩放 — 正确性+稳定性
8. **P1-5** Kahan buffer pin_memory — 消除同步 PCIe
9. **P1-6** block swap 独立 stream — 减少同步 stall
10. **P1-7** per-param → group optimizer — kernel launch -5x

### 补测试

11. **P2-6** LoRA remap/split round-trip 测试 — 防 P0-1 回归

---

## 已确认无问题的部分 (正面确认)

- ✅ **QKV 融合**: chunk 顺序与 cat 一致，零拷贝 view
- ✅ **AdaLN 融合 (非 LoRA 路径)**: 9 个向量顺序正确，block-diagonal 互逆
- ✅ **state_dict 权重重映射**: cat/split 精确互逆，有测试覆盖
- ✅ **RoPE dtype 上提**: 提前 cast + 条件 cast，无遗漏路径
- ✅ **学习率分组**: 融合后子串匹配仍正确
- ✅ **DataLoader fork 安全**: sqlite3 readonly 修复完整
- ✅ **设备处理**: 无硬编码 cuda:0
- ✅ **数值边界**: timestep_quantile 边界正确
- ✅ **融合操作零拷贝**: chunk/reshape 均 view
