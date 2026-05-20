# Anima 训练速度优化 — 待讨论方案

本文件记录暂未实施、需后续分析的优化项。已实施的 A/B/C(RoPE cos/sin 上提、AdaLN SiLU 去重、移除 RMSNorm autocast 装饰器)直接进了代码,见 `models/cosmos_predict2_modeling.py`。

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

## H. 用 `torch.compile` 包裹 TransformerBlock

### 现状

代码里没有任何 `torch.compile` 调用。整个 block forward 由 ~15 个小算子组成(3 个 layer_norm、3 个 adaln 投影、self-attn、cross-attn、mlp、若干 rearrange 与残差加法),每个算子都有独立的 Python dispatch + autograd graph 节点开销。block 在一个 size bucket 内 shape 固定,非常适合 compile。

### 优化思路

```python
# 在 to_layers() 或 enable_block_swap() 路径上,可选地:
for block in transformer.blocks:
    block.forward = torch.compile(block.forward, mode='reduce-overhead', fullgraph=False)
```

或者更稳的 `mode='default'` 配合 dynamic shapes 关闭。

### 预估收益

文献上 DiT 类模型 block 加 compile 通常能拿到 10–25% step time 削减(尤其在 bf16/fp8 small batch 情况下,kernel launch 开销占比大的时候)。对 Anima 这类 28-block × 大量小算子的结构,提升空间偏上限。

### 风险与待决问题

1. **与 Deepspeed pipeline parallel 的交互**:`train.py:564-578` 走的是 deepspeed 的 PipelineModule + activation checkpoint。compile 的 block 在被 deepspeed 切分到不同 stage 上时是否正确反向、是否能被 checkpoint 包裹,需要实测。已知风险点:
   - deepspeed pipeline 的 send/recv 不能在 compile 边界内。
   - `torch.utils.checkpoint(use_reentrant=True/False)` 与 compile 的兼容性:reentrant 在新版 PyTorch 上和 compile 配合较好,non-reentrant 历史上有坑。
2. **`unsloth_checkpoint`**:`activation_checkpointing='unsloth'` 走自定义路径(`utils/unsloth_utils.py`),与 compile 同时启用风险较高。
3. **Block-swap 兼容**:`ModelOffloader.wait_for_block`/`submit_move_blocks_forward` 在 `TransformerLayer.forward` 里调用(`cosmos_predict2.py:609-611`),compile 边界要选在 `block.forward` 而不是 `TransformerLayer.forward`,避免把异步 swap 拽进 graph。
4. **首次 compile 耗时**:Anima 28 blocks 各 compile 一次,首步可能要分钟级 warmup。可考虑只 compile 一个 block(因 28 个 block 结构同构,fx graph 应该缓存复用)。
5. **shape 抖动**:`size_bucket` 切换会触发 recompile;buckets 多时反而拖慢。需要确认训练里 bucket 切换频率,或先关闭 dynamic shape 强制每 bucket 一份 graph。
6. **fp8 / bnb 量化层**:目前 transformer_dtype 支持 float8,bnb-quantized 模块对 dynamo 的支持参差,需要白名单跳过。

### 落地步骤(草案)

1. 加 `[model] torch_compile = true/false` 配置项(默认 false),从 `model_config` 读取。
2. 在 `CosmosPredict2Pipeline.load_diffusion_model` 完成后,根据开关对 `self.transformer.blocks[i]` 做 `torch.compile(block, mode='default', fullgraph=False, dynamic=False)`。注意 compile 的是 block 实例,而非 `TransformerLayer.forward`(避免吞 offloader 异步操作)。
3. 在 docs 里加注:与 `activation_checkpointing='unsloth'`、`text_encoder_nf4` 等组合的已知限制。
4. 基线测试:不开 compile / 开 compile,跑 100 step,记录 step time、显存峰值、loss 曲线对照。
5. 跑过夜稳定性测试(>= 1 epoch)排查 graph recompile 抖动。

---

## 后续可顺手做的小项(参考,不在 G/H 范围)

- **D. LLM Adapter 的 (cos, sin) 缓存**(`models/llm_adapter.py:193-196`):6 层共用同一对 cos/sin,可按 seq_len 缓存。收益小,改动也小。
- **E. `padding_mask` 常量缓存**(`models/cosmos_predict2.py:556`):始终为零,可按 (B, H, W, dtype) 缓存或干脆移除 `concat_padding_mask` 通道(后者动模型结构,涉及 in_channels)。
- **F. `torch_attention_op` 用 `transpose` 代替 einops `rearrange`**(`models/cosmos_predict2_modeling.py:299-306`):batch=1 的 LoRA 训练下 einops Python 解析开销可见。
