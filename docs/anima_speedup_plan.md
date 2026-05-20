# Anima 训练速度优化 — 待讨论方案

本文件记录暂未实施、需后续分析的优化项。已实施的项目直接进了代码:
- A/B/C(RoPE cos/sin 上提、AdaLN SiLU 去重、移除 RMSNorm autocast 装饰器):见 `models/cosmos_predict2_modeling.py`。
- H(per-block `torch.compile`,opt-in,见下方):见 `models/cosmos_predict2.py:to_layers`。

适用范围:Anima 与 Cosmos-Predict2 共用的 `MiniTrainDIT`(`models/cosmos_predict2_modeling.py`)。Anima 配置见 `docs/supported_models.md` 的 Anima 节。

---

## G. 把每个 Block 的三组 AdaLN modulation Linear 融成一组 GEMM

### 现状

`TransformerBlockAdaLN.__init__`(`models/cosmos_predict2_modeling.py:991-1009`)给每个 block 建了三组独立的 modulation 投影:

```python
self.adaln_modulation_self_attn  = nn.Sequential(SiLU, Linear(D, lora_dim), Linear(lora_dim, 3*D))
self.adaln_modulation_cross_attn = nn.Sequential(SiLU, Linear(D, lora_dim), Linear(lora_dim, 3*D))
self.adaln_modulation_mlp        = nn.Sequential(SiLU, Linear(D, lora_dim), Linear(lora_dim, 3*D))
```

`use_adaln_lora=False` 时是 `Sequential(SiLU, Linear(D, 3*D))`。

每个 block forward 里这三组 Linear 输入相同(`silu_emb`,即 SiLU(emb_B_T_D))、输出分别 `3*D`,然后各自 chunk 成 (shift, scale, gate)。

### 优化思路

合并为单一 Linear:
- `use_adaln_lora=False`:`Linear(D, 9*D)`,一次 GEMM 出全部 9 个 (shift/scale/gate) × (self/cross/mlp)。
- `use_adaln_lora=True`:`Linear(D, lora_dim_total)` → `Linear(lora_dim_total, 9*D)`,两次更大的 GEMM。`lora_dim_total` 可以是 `3*lora_dim`(把三组的瓶颈维度也拼起来),也可以仍是 `lora_dim`(更激进,瓶颈共享)。

GEMM 在更大的矩阵上效率显著更高(尤其 D=2048 的 Anima),且省了两次 Python-level Linear 调用 + 两次 autograd graph 节点。

### 预估收益

- 每 block 节省 2 次 small-D Python-level Linear 调用,28 blocks × 28 step/s = 上千次/秒额外开销消除。
- 单次 9× 更大的 GEMM 比三次小 GEMM 在 H100/A100 上 throughput 通常高 10–20%(取决于 D 和 batch)。
- 在大 batch 或大序列下 Linear 本身已小,占比有限;在 LoRA 训练(batch=1)时收益更显著。

### 风险与待决问题

1. **state_dict 兼容性**:模块 key 会变。需要在 `CosmosPredict2Pipeline.load_diffusion_model`(`models/cosmos_predict2.py:259-305`)做参数重映射(把三个 Linear 权重 cat 到一个权重)。LoRA 保存路径(`save_adapter`,`save_model`)若被 LoRA 工具基于原 key 注入也需同步。
2. **PEFT/LoRA target modules**:`adapter_target_modules` 现含 `'Block'`(`cosmos_predict2.py:181`)。如果有用户依赖按 `adaln_modulation_self_attn` 名字做精细配置,需要决定是否暴露兼容别名。
3. **`get_param_groups` 学习率分组**:`mod_params` 的归类基于名字含 `.adaln_modulation`(`cosmos_predict2.py:460`),合并后只剩一个 modulation key,逻辑要更新(从代码上看这是同名前缀的子串匹配,只要新模块名仍包含 `adaln_modulation` 就行,但要复核 `cross_attn_lr`/`self_attn_lr`/`mlp_lr` 这套分别设定的能力会失效 —— 三组合一后无法再分头给学习率)。
4. **数值等价性**:合并 GEMM 在 bf16 下与三次独立 GEMM 数值上不严格相等(累加顺序不同),但量级一致,训练上等价。

### 落地步骤(草案)

1. 在 `TransformerBlockAdaLN.__init__` 加一个开关参数 `fused_adaln=True`(默认开),分支构造单一融合 Linear 模块。
2. 在 forward 里 `silu_emb` 之后做一次大投影,reshape 成 `(3, 3, B, T, D)` 然后切。
3. 在 `load_diffusion_model` 中检测 state_dict 是否含原始三组 key,若是则在加载时把三组权重 cat 起来塞进融合权重(`torch.cat([self_attn_w, cross_attn_w, mlp_w], dim=0)`)。保存时反向拆开以维持 ComfyUI 检查点格式。
4. 更新 `get_param_groups`:把 mod_params 的细分关闭或保留(若仍想支持分组,需要在融合层用 named buffer/parameter 分段管理 —— 不推荐,失去合并的意义)。
5. 跑一遍训练 + 推理对照测试,确认 loss 曲线不漂、ComfyUI 加载的 LoRA 推理正确。

---

## H. 用 `torch.compile` 包裹 TransformerBlock —— 已实施(opt-in)

实现位置:`models/cosmos_predict2.py:to_layers`(在所有 PEFT/LoRA 注入完成、`enable_block_swap` 之后)。

```python
if self.model_config.get('torch_compile', False):
    compile_mode = self.model_config.get('torch_compile_mode', 'default')
    compile_dynamic = self.model_config.get('torch_compile_dynamic', False)
    for block in transformer.blocks:
        block.forward = torch.compile(block.forward, mode=compile_mode, fullgraph=False, dynamic=compile_dynamic)
```

配置项:
- `torch_compile = true/false`(默认 `false`):总开关。
- `torch_compile_mode = 'default' | 'reduce-overhead' | 'max-autotune' | ...`(默认 `'default'`):传给 `torch.compile` 的 mode。`'default'` 最稳;`'reduce-overhead'` 启用 CUDA graphs 但与 block_swap、reentrant checkpoint 副作用兼容性更差;`'max-autotune'` warmup 极长。
- `torch_compile_dynamic = true/false`(默认 `false`):传给 `torch.compile` 的 `dynamic`。`false` 对单 size bucket 训练生成的图最优,但多 bucket 训练时每个新 (block × shape) 都会触发 recompile;多 bucket 训练应该设 `true`。

参数固定不暴露:
- **`fullgraph=False`**:`einops.rearrange` 的 Python 实现一定会触发 graph break,设 `true` 会让 compile 直接 raise,无意义暴露。
- **compile 边界:`Block.forward`**:外层 `TransformerLayer.forward` 里的 `offloader.wait_for_block` / `submit_move_blocks_forward` 留在 graph 外,异步 swap 不会被 dynamo 拽进 graph。这是个实现约束,不是用户选项。

文档见 `docs/supported_models.md` Anima 节。

### 已知限制(运行时会 print warning,不强制 block)

1. **`activation_checkpointing = 'unsloth'`**:`utils/unsloth_utils.py` 的 `unsloth_checkpoint` 被 `@torch._disable_dynamo` 装饰,compile 在被它包住的 block forward 上变成 no-op。
2. **`blocks_to_swap > 0`**:`ModelOffloader.swap_weight_devices_cuda` 每步原地改写 `module.weight.data`,dynamo guards 大概率被打破,会持续 recompile。
3. **`transformer_dtype = 'float8'`**:fp8 在 dynamo 下覆盖率仍在补,常见 graph break。
4. **多 size bucket 训练**:`dynamic=False`(默认)下每个新 shape × 每个 block 都会 recompile;buckets 多时设 `torch_compile_dynamic = true`,代价是图稍不优化。

### 待验证(没在 CI 跑过,需要用户实测)

1. 不开 / 开 compile,baseline vs compile 跑 100 step,记录 step time、显存峰值、loss 曲线对照。
2. 过夜稳定性测试(≥ 1 epoch),排查长跑 graph recompile 抖动。
3. `pipeline_stages > 1` 时 deepspeed pipeline 的 send/recv 是否仍在 compile 边界外(理论上是,因为 `TransformerLayer.forward` 才是 deepspeed 看到的层)。
4. Cosmos-Predict2 14B / Cosmos-Predict2 base 也走这条路径,但块数和维度不同,实际收益需要分档测。

---

## 后续可顺手做的小项(参考,不在 G/H 范围)

- **D. LLM Adapter 的 (cos, sin) 缓存**(`models/llm_adapter.py:193-196`):6 层共用同一对 cos/sin,可按 seq_len 缓存。收益小,改动也小。
- **E. `padding_mask` 常量缓存**(`models/cosmos_predict2.py:556`):始终为零,可按 (B, H, W, dtype) 缓存或干脆移除 `concat_padding_mask` 通道(后者动模型结构,涉及 in_channels)。
- **F. `torch_attention_op` 用 `transpose` 代替 einops `rearrange`**(`models/cosmos_predict2_modeling.py:299-306`):batch=1 的 LoRA 训练下 einops Python 解析开销可见。
