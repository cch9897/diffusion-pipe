# diffusion-pipe 审议问题修复计划 (v2)

> 基于 CONSOLIDATED_REVIEW.md (2026-06-17) + 3-Agent 交叉投票修正
> 全部 30 项已验证，30 项决议如下

---

## 交叉投票结果摘要

- ✅ 可直接执行: 11 项 (B3, B2, H5, B4, H4, H6, M2, M3, M5, H2)
- ⚠️ 需修正后执行: 7 项 (B1, B5, H3, M1, M4, M6, H9)
- ❌ 需重新设计: 2 项 (H1, H7)

---

## Phase 0 — 零风险速修 (预计 30min)

全部 ✅ 共识通过，无争议，可直接执行。

### B3. train.py:834 `'lora'` → `'adapter'`

- **文件**: `train.py:834`
- **改动**:
  ```python
  # Before
  communication_data_type = config['lora']['dtype'] if 'lora' in config else config['model']['dtype']
  # After
  communication_data_type = config['adapter']['dtype'] if 'adapter' in config else config['model']['dtype']
  ```
- **验证**: 用 examples/*.toml 跑 dry-run 确认走 adapter 分支。

### B2. LLM Adapter in-place → masked_fill

- **文件**: `models/cosmos_predict2.py:1023`
- **改动**:
  ```python
  # Before
  crossattn_emb[~t5_attn_mask.bool()] = 0
  # After
  crossattn_emb = crossattn_emb.masked_fill(~t5_attn_mask.bool().unsqueeze(-1), 0)
  ```
- **注意**: `unsqueeze(-1)` 对齐最后一维。`t5_attn_mask` shape `(B, T)`，`crossattn_emb` shape `(B, T, D)`。
- **验证**: `llm_adapter_lr > 0` 时前向+反向不报 inplace error。

### H5. has_inf_or_nan 溢出修复

- **文件**: `optimizers/generic_optim.py:23-25`
- **改动** (A2 建议用 `dtype=` 参数，比 `.float().sum()` 更省内存——不创建完整 float 副本):
  ```python
  # Before
  def has_inf_or_nan(x):
      s = x.sum()
      return s.isinf() or s.isnan()
  # After
  def has_inf_or_nan(x):
      s = x.sum(dtype=torch.float32)
      return s.isinf() or s.isnan()
  ```

### B4. cache.py add() 定期 commit

- **文件**: `utils/cache.py:139-163`
- **改动**: 在 `add()` 末尾加定期提交 + A3 建议的字节阈值回退:
  ```python
  # 在 self.shard_index += 1 之后
  if self.shard_index % 1000 == 0 or self.shard_file.tell() > 500_000_000:
      self.con.commit()
  ```
- **验证**: 缓存生成中途 kill 进程，重启后确认已 commit 的记录存在。

### H4. CUDA Event 复用

- **文件**: `optimizers/generic_optim.py:504-505`
- **改动**:
  ```python
  # Before
  self._sync_event = torch.cuda.Event()
  self._sync_event.record()
  self._sync_event.synchronize()
  # After
  if not hasattr(self, '_sync_event') or self._sync_event is None:
      self._sync_event = torch.cuda.Event()
  self._sync_event.record()
  self._sync_event.synchronize()
  ```

### H6. _reopen_caches_readonly 循环引用防护

- **文件**: `utils/dataset.py:1309-1319`
- **改动**:
  ```python
  def _reopen_caches_readonly(obj, visited=None):
      if visited is None:
          visited = set()
      obj_id = id(obj)
      if obj_id in visited:
          return
      visited.add(obj_id)
      from utils.cache import Cache
      if isinstance(obj, Cache):
          obj.init_readonly()
      elif hasattr(obj, '__dict__'):
          for v in obj.__dict__.values():
              _reopen_caches_readonly(v, visited)
      elif isinstance(obj, (list, tuple)):
          for item in obj:
              _reopen_caches_readonly(item, visited)
  ```

---

## Phase 1 — 正确性核心 (预计 2h)

⚠️ 需修正后执行。以下为交叉投票修正后的方案。

### B1. Kahan 求和修复 (2 处，分别编写)

#### B1-a. generic_optim.py:485-496

- **当前代码**:
  ```python
  shift = state['shift'].to(p.device, non_blocking=True)
  shift.add_(update)          # BUG
  p.grad.copy_(p.detach())
  p.add_(shift)
  shift.add_(p.grad.sub_(p))
  state['shift'] = shift.to(kahan_buffer_device)
  ```
- **修复** (标准 Neumaier 补偿):
  ```python
  shift = state['shift'].to(p.device, non_blocking=True)
  y = update - shift
  t = p.detach() + y
  new_shift = (t - p.detach()) - y
  p.copy_(t)
  shift.copy_(new_shift)
  state['shift'] = shift.to(kahan_buffer_device)
  ```
- **关键**: generic_optim 中 `update` 在进入 Kahan 块之前已被取反（外层逻辑），此处 `update` 语义为 "要加到参数上的增量"，直接套用 Neumaier。

#### B1-b. automagic.py:308-318 (⚠️ 不能套用 generic_optim 模板)

- **当前代码**:
  ```python
  if p.dtype == torch.bfloat16:
      # Kahan summation for bfloat16
      update.mul_(-1)                                    # ← 取反在 Kahan 块内部
      if weight_decay_update is not None:
          update.add_(weight_decay_update)
      shift = state['shift']                             # ← 直接引用，无 .to()
      shift.add_(update)                                 # BUG
      grad.copy_(p.detach())                             # ← 用 grad 局部变量，非 p.grad
      p.add_(shift)
      shift.add_(grad.sub_(p))
  ```
- **A2 指出的 3 处差异**:
  1. `update.mul_(-1)` 在 Kahan 块内部，取反后 update = -原始梯度 + weight_decay
  2. 无 `kahan_buffer_device`，shift 直接引用 `state['shift']`，不跨设备传输
  3. 用 `grad` 局部变量做临时 buffer，非 `p.grad`
- **修复** (适配上述差异):
  ```python
  if p.dtype == torch.bfloat16:
      update.mul_(-1)
      if weight_decay_update is not None:
          update.add_(weight_decay_update)
      # Neumaier compensated summation
      shift = state['shift']
      y = update - shift
      t = p.detach() + y
      new_shift = (t - p.detach()) - y
      p.copy_(t)
      shift.copy_(new_shift)
  ```
- **验证**: bf16 参数训练 100 步，对比修复前后 loss 曲线。两个优化器分别测试。

### B5. cache.py weights_only 回退安全

- **文件**: `utils/cache.py:35-43`
- **A1 修正**: 新版 PyTorch 已有 `weights_only` 时不应回退到 pickle。仅保留 TypeError (旧 PyTorch 兼容)。
- **改动**:
  ```python
  # Before
  try:
      item = torch.load(buffer, map_location='cpu', weights_only=True)
  except TypeError:
      item = torch.load(buffer, map_location='cpu')
  except Exception:
      item = torch.load(buffer, map_location='cpu')
  # After
  try:
      item = torch.load(buffer, map_location='cpu', weights_only=True)
  except TypeError:
      # Older PyTorch without weights_only parameter
      item = torch.load(buffer, map_location='cpu')
  # 不再捕获其他异常: 数据损坏/磁盘错误应向上传播，不静默回退 pickle
  ```
- **验证**: 正常缓存数据加载无异常；损坏数据不再静默回退，直接抛出。

### H3. CUDA Stream 复用

- **文件**: `utils/offloading.py:71` (⚠️ A1 指出报告中误写为 `optimizers/offloading.py`，正确路径是 `utils/offloading.py`)
- **当前**: 每次 `swap_weight_devices_cuda` 调用 `stream = torch.cuda.Stream()` 新建。
- **修复方案**: 模块级复用:
  ```python
  _swap_stream = None

  def swap_weight_devices_cuda(...):
      global _swap_stream
      if _swap_stream is None:
          _swap_stream = torch.cuda.Stream()
      stream = _swap_stream
      with torch.cuda.stream(stream):
          ...
  ```

---

## Phase 2 — 中级稳健性 (预计 2h)

### M2. pool.close() 缺 pool.join()

- **文件**: `utils/dataset.py:158`
- **改动**:
  ```python
  pool.close()
  pool.join()  # 新增
  cache.finalize_current_shard()
  ```

### M3. cache.py open_files LRU 限制

- **文件**: `utils/cache.py:30, 82`
- **改动**: 用 `collections.OrderedDict` 替代 dict:
  ```python
  from collections import OrderedDict
  self.open_files = OrderedDict()
  # _get_item 中:
  if shard_id in self.open_files:
      self.open_files.move_to_end(shard_id)
  elif len(self.open_files) >= 64:
      _, old_f = self.open_files.popitem(last=False)
      old_f.close()
  self.open_files[shard_id] = open(self.path / f'shard_{shard_id}.bin', 'rb')
  ```

### M1. tarfile_map 句柄 LRU (⚠️ A3 要求补充调用位置 + LRU 实现)

- **文件**: `utils/dataset.py:738-835`
- **调用位置分析**: `tarfile_map` 是 `_metadata_map_fn` 内的闭包变量 (line 738)，被 `fn` (line 740) 捕获。`fn` 作为 map 函数经 `pool.imap` 在 worker 进程中执行 (line 153)。每个 worker 有自己的 `tarfile_map`，互不共享。
- **问题**: 长时间运行中 tarfile_map 无限增长，每个 worker 累积文件描述符。
- **方案**: 复用 M3 的 LRU 模式，在闭包内部限制:
  ```python
  def _metadata_map_fn(self):
      from collections import OrderedDict
      tarfile_map = OrderedDict()
      MAX_OPEN_TARS = 32

      def fn(example):
          ...
          if image_spec[0] is None:
              tar_f = None
              filepath_or_file = str(image_file)
          else:
              tar_filename = image_spec[0]
              if tar_filename in tarfile_map:
                  tarfile_map.move_to_end(tar_filename)
              elif len(tarfile_map) >= MAX_OPEN_TARS:
                  _, old_tar = tarfile_map.popitem(last=False)
                  old_tar.close()
              if tar_filename not in tarfile_map:
                  tarfile_map[tar_filename] = tarfile.TarFile(tar_filename)
              tar_f = tarfile_map[tar_filename]
              filepath_or_file = tar_f.extractfile(str(image_file))
          ...
      return fn
  ```
- **验证**: 缓存含 >32 个 tar 文件的 dataset，监控 worker 进程 fd 数不超限。

### M4. cache.py items 表加主键 (⚠️ A3 要求加迁移逻辑)

- **文件**: `utils/cache.py:66, 46-85`
- **问题**: `CREATE TABLE IF NOT EXISTS` 对已有数据库不生效，旧库无主键。
- **方案**: 在 `init()` 中检测 schema 并迁移:
  ```python
  # items 表创建
  self.con.execute('CREATE TABLE IF NOT EXISTS items(shard INT, shard_index INT, PRIMARY KEY(shard, shard_index))')

  # 迁移: 检测旧表无主键时重建
  cols = self.con.execute('PRAGMA table_info(items)').fetchall()
  has_pk = any(col[5] for col in cols)  # col[5] = pk flag
  if cols and not has_pk:
      print('[CACHE] Migrating items table to add primary key')
      self.con.execute('ALTER TABLE items RENAME TO items_old')
      self.con.execute('CREATE TABLE items(shard INT, shard_index INT, PRIMARY KEY(shard, shard_index))')
      self.con.execute('INSERT OR IGNORE INTO items SELECT * FROM items_old')
      self.con.execute('DROP TABLE items_old')
      self.con.commit()
  ```
- **验证**: 用旧格式缓存数据库启动，确认迁移成功且无重复记录。

### M6. cache.py clear() 异常处理 (⚠️ A3 要求分两阶段)

- **文件**: `utils/cache.py:88-94`
- **A3 修正**: 分两阶段——先删 db，失败则中止；再删 bin。
- **改动**:
  ```python
  def clear(self):
      self.con.close()
      # Phase 1: 删除数据库，失败则中止（bin 文件保留，可手动清理）
      try:
          os.remove(self.metadata_db)
      except OSError as e:
          print(f'[CACHE] FATAL: could not remove database {self.metadata_db}: {e}')
          raise
      # Phase 2: 删除 bin 文件，逐个处理，部分失败不中止
      for bin_path in self.path.glob('*.bin'):
          try:
              os.remove(bin_path)
          except OSError as e:
              print(f'[CACHE] Warning: could not remove {bin_path}: {e}')
      self.init()
  ```

### M5. cache.py init_readonly 竞争

- **文件**: `utils/cache.py:132-136`
- **现状**: `init_readonly()` 在 `torch.utils.data` worker fork 后调用，每个 worker 有独立内存空间，实际竞争风险低。当前 `uri=True, mode=ro` 已避免写锁竞争。
- **方案**: 加注释说明设计意图，暂不改:
  ```python
  def init_readonly(self):
      """Reopen SQLite in read-only mode.

      Called from _worker_init_fn after DataLoader fork. Each worker gets
      its own copy of the Cache object (copy-on-write), so there is no
      cross-worker contention on self.con. Read-only URI mode avoids
      SQLite write lock contention entirely.
      """
      self.con.close()
      self.con = sqlite3.connect(
          f'file:{self.metadata_db}?mode=ro', uri=True
      )
      self.open_files = {}
  ```

### H2. lora_A 有损平均补 warning

- **文件**: `models/cosmos_predict2.py:281, 309-311`
- **改动**: KV 和 AdaLN 路径补 warning，与 QKV 路径 (line 251-254) 一致:
  ```python
  # KV lora_A (line 279-281)
  if lora_type == 'lora_A':
      warnings.warn(
          f'Remapping legacy KV LoRA: averaging 2 lora_A tensors for {prefix}. '
          f'This is lossy (rank 2R -> R). Consider retraining with fused kv_proj.'
      )
      new_sd[fused_key] = (k_w + v_w) / 2.0

  # AdaLN lora_A (line 307-311)
  if lora_type == 'lora_A':
      warnings.warn(
          f'Remapping legacy AdaLN LoRA: averaging 3 lora_A tensors for {prefix}. '
          f'This is lossy (rank 3R -> R). Consider retraining with fused adaln_modulation.'
      )
      new_sd[fused_key_1] = sum(l1_list) / 3.0
      new_sd[fused_key_2] = sum(l2_list) / 3.0
  ```

---

## Phase 3 — 暂缓 / 需重新设计

### H1. AdaLN 块对角参数化重构 (❌ 需重新设计)

- **A1 发现**: v1 计划将 `adaln_modulation_1` 瓶颈维度从 `3*adaln_lora_dim` 降为 `adaln_lora_dim`，容量缩水 3 倍。必须保持 `3*adaln_lora_dim`。
- **正确设计**: 只拆 `adaln_modulation_2`，不拆 `adaln_modulation_1`:
  ```python
  # __init__ (cosmos_predict2_modeling.py:1037-1039)
  # Before:
  adaln_lora_dim_total = 3 * adaln_lora_dim
  self.adaln_modulation_1 = nn.Linear(x_dim, adaln_lora_dim_total, bias=False)  # D → 3R (保持不变)
  self.adaln_modulation_2 = nn.Linear(adaln_lora_dim_total, 9 * x_dim, bias=False)  # 3R → 9D (要拆)

  # After:
  adaln_lora_dim_total = 3 * adaln_lora_dim
  self.adaln_modulation_1 = nn.Linear(x_dim, adaln_lora_dim_total, bias=False)  # D → 3R (不变)
  # 拆为 3 个独立 Linear(R, 3D)，每个分支独立，无块对角约束
  self.adaln_modulation_2_self_attn = nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False)
  self.adaln_modulation_2_cross_attn = nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False)
  self.adaln_modulation_2_mlp = nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False)
  ```
- **前向** (cosmos_predict2_modeling.py:1078-1079):
  ```python
  # Before:
  mod_9D = self.adaln_modulation_2(self.adaln_modulation_1(silu_emb))

  # After:
  mid = self.adaln_modulation_1(silu_emb)  # (B, T, 3R)
  R = self.adaln_lora_dim
  mid_self = mid[..., :R]
  mid_cross = mid[..., R:2*R]
  mid_mlp = mid[..., 2*R:]
  mod_9D = torch.cat([
      self.adaln_modulation_2_self_attn(mid_self),
      self.adaln_modulation_2_cross_attn(mid_cross),
      self.adaln_modulation_2_mlp(mid_mlp),
  ], dim=-1)  # (B, T, 9*D)
  ```
- **影响**:
  - `_remap_state_dict_keys` (cosmos_predict2.py:85-126): 组装逻辑改为直接赋值 3 个独立权重
  - `_split_state_dict_keys` (cosmos_predict2.py:172-187): 拆分逻辑改为直接读取 3 个独立权重
  - LoRA remap (cosmos_predict2.py:287-318): 需适配新 key 名
  - checkpoint 格式变化: 旧 `adaln_modulation_2.weight` (9D×3R) 需拆为 3 个 (3D×R)
  - 迁移: `_remap_state_dict_keys` 中检测旧格式 `adaln_modulation_2.weight`，按块对角提取 3 个对角块
- **验证**: load → train 1 step → save → reload 往返一致性测试。
- **状态**: 暂缓，待设计 review 后独立分支实现。

### H7. mp.Queue 跨进程广播 (❌ 需重新设计)

- **A3 发现**: v1 方案 B 自相矛盾——`torch.multiprocessing.Queue()` 同样无法跨进程广播。
- **根因**: 任何 Python 级 Queue 对象都无法经 `broadcast_object_list` (pickle + broadcast) 在独立进程间共享。Manager.Queue 的代理引用在 DeepSpeed 进程组初始化不同步时可能失效。
- **重新设计方案**: 消除 Queue 广播需求，改为 rank-0 单进程驱动 + dist 通信协调:
  ```
  方案: 消除 broadcast_object_list 对 Queue 的依赖

  1. rank 0 创建 Manager.Queue()，非 rank 0 不创建
  2. 用 dist.broadcast_object_list 只广播一个 int 哨兵 (而非 Queue 对象)
     - 哨兵 = 0: rank 0 已完成 Manager 初始化
     - 非 rank 0 收到哨兵后进入等待循环
  3. rank 0 将任务通过 Manager.Queue 分发
     - 但 Manager.Queue 代理无法跨进程传递...

  → 根本问题: Manager.Queue 的代理依赖共享 Manager 进程，
    而 DeepSpeed 各 rank 是独立 spawn 的进程，不共享 Manager。

  最终方案: 不用 Manager.Queue 跨 rank 通信。
  - rank 0 的 cache 子进程直接处理所有数据预处理
  - 各 rank 通过 torch.distributed send/recv 或 broadcast_object_list
    逐条接收任务 (任务本身是可序列化的 dict，不含 Queue 引用)
  - 或: 各 rank 独立读取任务列表 (从共享文件或 dist.broadcast)，
    按rank 分配任务区间，无需跨进程 Queue
  ```
- **状态**: 暂缓，需实际 DeepSpeed 多 GPU 环境验证。当前单 GPU 训练不受影响。

### H9. gradient_release add_ monkeypatch (⚠️ 仅文档化)

- **文件**: `train.py:709-715`
- **A2 判定**: 这不是修复，只是文档化。
- **短期**: 加详细注释 + 作用域说明:
  ```python
  # WARNING: This monkeypatch replaces add_ on ALL parameters globally.
  # It bypasses autograd in-place detection by writing to .data directly.
  # Only safe when:
  # 1. gradient_release mode is active (one optimizer step per micro-batch)
  # 2. No other code path calls p.add_() expecting autograd tracking
  # This is intentionally hacky — see long-term ticket for replacement plan.
  ```
- **长期**: 创建独立工单追踪 `register_hook` 替代方案，不在本次修复中实现。

---

## Phase 4 — 清理 (预计 1h)

### M7. DotProductAttention import

- **文件**: `models/cosmos_predict2_modeling.py:424`
- **改动**:
  ```python
  # import 区
  try:
      from transformer_engine.pytorch import DotProductAttention
  except ImportError:
      DotProductAttention = None
  # __init__ 中
  if self.backend == "transformer_engine":
      if DotProductAttention is None:
          raise ImportError("transformer_engine is required for TE backend")
      self.attn_op = DotProductAttention(...)
  ```

### M8. unsloth 强制覆盖文档化

- **文件**: `train.py:103-104`
- **改动**: 加注释说明为何强制 `reentrant_activation_checkpointing = True`。

### M10. reentrant checkpoint 检测

- **文件**: `utils/offloading.py:287-289`
- **方案**: 维持现状但加注释说明 `is_grad_enabled()` 的局限，或后续改用显式 flag。

### L1-L6. Lint + 别名 + 废弃 API

- L1: `KEEP_IN_HIGH_PRECISION` 精确匹配 (cosmos_predict2.py:33, 684)
- L2: `llm_adapter.py` RMSNorm/RoPE 统一 (低优先级)
- L3: `cosmos_predict2_modeling.py:211` fused RoPE assert (加注释或实现)
- L4: `common.py:14-21` DTYPE_MAP 补 `fp16`/`fp32`/`bf16` 别名
- L5: `patches.py:20-23` `from torch._six import inf` → `from torch import inf`
- L6: `ruff check --fix` 自动修复 F401/E712/F841/E702/E741/F402/E711

---

## 执行顺序

```
批次 1: Phase 0 (B3 + B2 + H5 + B4 + H4 + H6)
         6 项 ✅ 全票通过 → commit → push

批次 2: Phase 1 (B1-a + B1-b + B5 + H3)
         4 项 ⚠️ 修正后执行 → commit → bf16 回归测试 → push

批次 3: Phase 2 (M1 + M2 + M3 + M4 + M6 + M5 + H2)
         7 项 → commit → push

批次 4: Phase 4 (M7 + M8 + M10 + L1-L6)
         清理 → commit → push

暂缓:
  H1: 需重新设计瓶颈维度 → 独立分支
  H7: 需重新设计 Queue 方案 → 需多 GPU 环境验证
  H9: 仅文档化，长期工单追踪 register_hook
```

---

## 验证策略

每个批次完成后:

1. **语法检查**: `python -c "import ast; ast.parse(open(f).read())"` 或 ruff
2. **import 检查**: `python -c "from models.cosmos_predict2 import *"` 等
3. **回归测试**: 至少跑 10 步 dry-run 训练确认不崩
4. **diff 审查**: 检查无密钥/IP/路径泄露
5. **commit + push**

---

*v2 更新: 2026-06-17 | 整合 3-Agent 交叉投票全部修正*
