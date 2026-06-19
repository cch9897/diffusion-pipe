# PLAN-B: fp32 Update Computation for GenericOptim (bf16 params)

## Problem Recap

- Model params are bf16. GenericOptim computes the entire update (lines 462-503) in bf16.
- bf16 epsilon at magnitude ~1.0 = 0.0078. With lr=6-8e-6, per-step updates are ~1300x below the bf16 quantization threshold.
- Kahan summation is self-defeating: the Kahan `shift` buffer is also bf16, so sub-threshold compensation values get quantized to zero.
- Result: most parameters never change; loss cycles with dataset epoch period.

## Design

**Core idea**: Keep model parameters in bf16 (no master weights → zero VRAM overhead for a second parameter copy). But compute the *update* tensor in fp32 and store the Kahan `shift` buffer in fp32 so that sub-bf16-threshold compensation CAN accumulate across steps. When enough compensation accrues (exceeds ~0.008), it finally "tips" the bf16 parameter.

### Key decisions

| What | Old | New | Rationale |
|------|-----|-----|-----------|
| `update` tensor dtype | bf16 (inherits from param) | fp32 | Preserve sub-bf16 values during update computation |
| Kahan `shift` buffer dtype | bf16 | fp32 | Allow compensation to accumulate below bf16 threshold |
| Numerator/denominator during update | bf16 arithmetic | fp32 (explicit `.float()`) | Preserve update precision |
| Momentum states (`exp_avg`, `exp_avg_sq`) | bf16 | **Unchanged (bf16)** | Momentum updates are O(0.1 * grad) ~ O(0.1), well above bf16 epsilon |
| Model params | bf16 | **Unchanged (bf16)** | No VRAM cost for master weights |
| Final write to param | bf16 truncation + Kahan | Stochastic rounding (optional) or simple truncation with fp32 Kahan catch | fp32 Kahan alone is sufficient |

---

## Exact Code Diff

### Change 1: Add import (top of file)

```diff
--- a/optimizers/generic_optim.py
+++ b/optimizers/generic_optim.py
@@ -13,6 +13,7 @@ import torch
 from torch.optim import Optimizer
 
 from transformers.utils.versions import require_version
+from .optimizer_utils import copy_stochastic
 from deepspeed import comm as dist
 from deepspeed.accelerator import get_accelerator
```

### Change 2: Create update tensor in fp32 (line 462)

```diff
-                update = torch.zeros_like(p)
+                update = torch.zeros_like(p, dtype=torch.float32)
```

### Change 3: Compute update in fp32 (lines 464-477)

Convert numerator/denominator to fp32 for the update computation:

```diff
                 # step
                 if denominator is None:  # no adaptive step size
-                    update.add_(numerator, alpha=-step_size)
+                    update.add_(numerator.float(), alpha=-step_size)
                 elif self.second_moment_type in ('ema', 'factored'):  # standard adam
-                    update.addcdiv_(numerator, denominator, value=-step_size)
+                    update.addcdiv_(numerator.float(), denominator.float(), value=-step_size)
                 elif self.second_moment_type == "sn":  # subset norm requires broadcast division
                     if "subset_size" in group and group["subset_size"] != "heuristics":
-                        norm_grad = (numerator.view(state["subset_shape"]) / denominator).reshape(p.shape)
+                        norm_grad = (numerator.float().view(state["subset_shape"]) / denominator.float()).reshape(p.shape)
                         update.add_(norm_grad, alpha=-step_size)
                     else:  # broadcast division is default for heuristics and non-subset-norm modules
-                        update.addcdiv_(numerator, denominator, value=-step_size)
+                        update.addcdiv_(numerator.float(), denominator.float(), value=-step_size)
                 else:
                     raise ValueError(...)
```

### Change 4: Weight decay in fp32 (line 480-481)

```diff
                 if group["weight_decay"] > 0.0:
-                    update.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
+                    update.add_(p.float(), alpha=(-group["lr"] * group["weight_decay"]))
```

### Change 5: Rewrite Kahan summation with fp32 shift buffer (lines 485-501)

The core rewrite. The old code uses `p.grad` as a bf16 temp buffer to avoid allocation. The new code uses the fp32 `shift` buffer itself as the accumulation temp (no extra allocation).

```diff
+                # Apply update to parameter
                 if p.dtype == torch.bfloat16:
-                    # In-place fused Kahan compensated summation for bfloat16.
-                    # Zero temporary allocations — arithmetic is hidden by memory bandwidth.
-                    # Uses the opposite-sign convention (shift holds -compensation),
-                    # bit-identical to Neumaier but without the 3-temp-tensor overhead.
+                    # fp32 Kahan compensated summation with bf16 parameter.
+                    # The shift buffer is fp32 so sub-bf16-threshold compensation
+                    # can accumulate across steps. The shift buffer doubles as the
+                    # fp32 accumulation temp to avoid extra allocations.
                     if 'shift' not in state:
-                        state['shift'] = torch.zeros_like(p)
+                        state['shift'] = torch.zeros_like(p, dtype=torch.float32)
                     shift = state['shift'].to(p.device, non_blocking=True)
-                    shift.add_(update)                 # shift = comp + update
-                    # Reuse p.grad as temp buffer (grad already consumed into update
-                    # above). Avoids allocating a dedicated _temp tensor the size of p,
-                    # which costs 4 GB on a 2B-param model and causes OOM on 32 GB cards.
-                    p.grad.copy_(p.detach())           # reuse p.grad as temp buffer
-                    p.add_(shift)                      # bf16-rounded addition
-                    shift.add_(p.grad.sub_(p))         # recover rounding error: comp = -(rounding error)
-                    # TODO: non_blocking=True here causes CUDA error on first step after checkpoint save.
+                    # Kahan: shift holds -compensation (opposite-sign convention).
+                    # Step 1: shift = -comp + update  (fp32)
+                    shift.add_(update)
+                    # Step 2: shift = p_fp32 + (-comp + update) = exact fp32 result
+                    shift.add_(p.float())
+                    # Step 3: truncate to bf16 and write back to parameter
+                    # (Option A: simple truncation — Kahan catches the error)
+                    p_bf16 = shift.bfloat16()
+                    p.copy_(p_bf16)
+                    # Step 4: recover rounding error for next step
+                    # shift = exact_fp32 - bf16_truncated = +rounding_error
+                    shift.sub_(p_bf16.float())
+                    # Convert to -rounding_error for next step's compensation
+                    shift.neg_()
                     state['shift'] = shift.to(kahan_buffer_device)
+                elif p.dtype == torch.float16:
+                    # Cast update to fp16 for fp16 params
+                    p.add_(update.half())
                 else:
+                    # fp32 params: direct addition
                     p.add_(update)
```

### Change 6 (optional enhancement): Use `copy_stochastic` instead of simple truncation

Replace the bf16 truncation with stochastic rounding for unbiased parameter updates:

```diff
-                    p_bf16 = shift.bfloat16()
-                    p.copy_(p_bf16)
+                    # Stochastic rounding: unbiased, prevents systematic drift
+                    # Uses bit-level int32 manipulation from optimizer_utils
+                    p_fp32_rounded = shift.clone()  # copy_stochastic writes to target
+                    copy_stochastic(p, p_fp32_rounded)
+                    p_bf16 = p.float()  # read back for rounding error computation
```

> **Trade-off**: Stochastic rounding requires `torch.randint_like` call (RNG overhead) and a clone of the shift tensor. It gives unbiased rounding which is theoretically better but empirically may not matter if the fp32 Kahan shift already handles accumulation correctly. **Recommended to start without it** and add only if systematic parameter drift is observed.

---

## VRAM Impact Analysis

| Component | Old size (per param) | New size | Delta | Persistent? |
|-----------|---------------------|----------|-------|-------------|
| Kahan `shift` buffer | N × 2 bytes (bf16) | N × 4 bytes (fp32) | +2 bytes/param | Yes |
| `update` tensor | N × 2 bytes (bf16) | N × 4 bytes (fp32) | +2 bytes/param | No (freed per iteration) |
| Temp `p_bf16` | N/A (reused p.grad) | N × 2 bytes | +2 bytes/param | No (freed per iteration) |
| Momentum states | Unchanged | Unchanged | 0 | — |

**Example: 2B-param model**
- Persistent overhead: 2B × 2 bytes = **+4 GB** GPU memory for fp32 shift buffers.
- **Mitigation**: If `kahan_buffer_offload=True` (already supported), shift buffers live on CPU → 0 GPU overhead.
- Peak temporary overhead (within one `step()` call for a single param): ~2 × largest_param_size × 4 bytes. For a ~500M-element embedding: ~4 GB peak. This is freed before processing the next parameter.

**With `kahan_buffer_offload=True`: effectively 0 persistent GPU overhead.**

---

## Impact on Existing Paths

| Path | Interaction | Safe? |
|------|------------|-------|
| Standard Adam/EMA | update uses fp32 numerator/denominator | ✅ |
| Subset norm (SN) | Broadcast division in fp32 | ✅ |
| Muon | `denominator=None`, uses numerator path — `.float()` conversion fine | ✅ |
| Adamuon | `denominator=None` after orthogonalization — same as muon | ✅ |
| Normuon | `denominator=None`, numerator already processed — `.float()` conversion fine | ✅ |
| Automagic LR | `numerator.mul_(automagic_lr)` modifies numerator in-place before fp32 conversion — fine | ✅ |
| CPU offload / Kahan buffer offload | `kahan_buffer_device` unchanged; shift is now fp32 on that device | ✅ |
| Weight decay | Now computed in fp32 using `p.float()` | ✅ |
| Factored second moment | denominator is fp32-division compatible | ✅ |
| Bias correction | `step_size` is a scalar (fp64 Python float) — unaffected | ✅ |
| fp32 params | `p.add_(update)` with fp32 update — direct, no precision loss | ✅ |
| fp16 params | New `elif` branch casts update to fp16 before addition | ✅ |

---

## Verification Plan

### Phase 1: Unit Test (precision smoke test)

```python
import torch
from optimizers.generic_optim import GenericOptim

def test_bf16_update_actually_changes_params():
    """Verify bf16 params change with tiny updates (lr=8e-6)."""
    torch.manual_seed(42)
    p = torch.nn.Parameter(torch.ones(1000, dtype=torch.bfloat16))
    opt = GenericOptim([p], lr=8e-6, betas=(0.0, 0.0), weight_decay=0.0,
                       momentum_type="none", second_moment_type="none")

    # Record initial values
    initial = p.detach().clone()

    # Run 1000 steps with alternating-sign gradients of magnitude ~1
    for i in range(1000):
        p.grad = torch.full_like(p, 1.0 if i % 2 == 0 else -1.0)
        opt.step()

    # Parameters should have changed measurably
    max_change = (p - initial).abs().max().item()
    print(f"Max parameter change after 1000 steps: {max_change}")
    assert max_change > 0.001, f"Parameters barely changed: {max_change}"
    print("PASS: bf16 params changed measurably")

def test_kahan_accumulates_sub_threshold():
    """Verify Kahan shift accumulates sub-bf16-threshold corrections."""
    torch.manual_seed(42)
    p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    opt = GenericOptim([p], lr=1e-3, betas=(0.0, 0.0),
                       momentum_type="none", second_moment_type="none")

    # Tiny gradient that won't change bf16 param in one step
    p.grad = torch.tensor([0.001])  # update = -1e-6, below bf16 eps
    for _ in range(100):
        p.grad = torch.tensor([0.001])
        opt.step()

    # After 100 steps, Kahan shift should have accumulated enough
    assert p.item() != 1.0, "Kahan failed to accumulate"
    print(f"PASS: Kahan accumulated: {p.item()}")

if __name__ == "__main__":
    test_bf16_update_actually_changes_params()
    test_kahan_accumulates_sub_threshold()
```

### Phase 2: Training Smoke Test

1. Run a short training (100 steps) with the patched optimizer.
2. Monitor:
   - **Parameter change**: Sample ~5 parameters and log their values every 10 steps. Verify they change monotonically (not cycling back).
   - **Gradient norm**: Should not cycle with epoch period.
   - **Loss**: Should decrease or at least not show epoch-periodic cycling.
   - **Kahan shift magnitude**: Log `state['shift'].abs().mean()` for a few params — should be non-zero.

3. Compare against **Plan A** (master weights in fp32, param copy in bf16) if implemented, or against a "no fix" baseline.

### Phase 3: Memory Profiling

```python
import torch.cuda
torch.cuda.reset_peak_memory_stats()
# ... run a training step ...
peak = torch.cuda.max_memory_allocated()
print(f"Peak GPU memory: {peak / 1e9:.2f} GB")
```

Compare peak memory before and after the patch. Verify increase is as predicted (~largest_param * 4 bytes for temporary fp32).

### Phase 4: Speed Benchmark

Run 100 training steps with `torch.cuda.Event` timing. Compare steps/sec before and after.

Expected: minimal slowdown (<5%). The fp32 arithmetic is on small tensors (per-param), and the `.float()` / `.bfloat16()` conversions are bandwidth-bound operations that are fast. The fp32 Kahan arithmetic is 4 operations vs the old bf16 Kahan's 4 operations. The main new overhead is `p.float()` allocation (once per param per step).

### Phase 5: Full Training Run

Run a full training epoch and verify:
- Loss converges (no cycling)
- Checkpoint save/load works (shift buffers are fp32 now, need to verify `load_state_dict` compatibility)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| fp32 shift 2x memory if not offloaded | High | Medium | Set `kahan_buffer_offload=True`; document this requirement |
| `load_state_dict` incompatibility | Low | High | Test checkpoint resume explicitly; old bf16 checkpoints will load fp32 shift from zeros |
| Numerical difference changes training dynamics | Medium | Medium | Compare loss curves with master-weights approach |
| Stochastic rounding RNG overhead | Low | Low | Make it optional; start without it |
| fp16 param path untested | Low | Low | Most users train in bf16 or fp32; fp16 path is a simple cast |

---

## Alternative: Stochastic Rounding Instead of fp32 Kahan

If we use stochastic rounding for the final write, we could keep the Kahan shift in fp32 but use unbiased rounding:

```python
# Instead of:
p_bf16 = shift.bfloat16()   # truncation
p.copy_(p_bf16)

# Use:
copy_stochastic(p, shift)   # unbiased stochastic rounding
```

This is simpler code-wise (the Kahan shift still needs to be fp32 to accumulate the fp32 update). The advantage: no systematic truncation bias. The disadvantage: RNG call per parameter, slightly slower.

**Recommendation**: Start with truncation + fp32 Kahan. Add stochastic rounding only if parameter histograms show systematic drift.

---

## Summary

This PLAN-B is a **minimal, surgical change** (touches only the step() method in generic_optim.py, ~30 lines changed). It:

1. **Solves the root cause**: fp32 update computation eliminates quantization loss during the critical `update = -lr * numerator / denominator` step.
2. **Preserves bf16 params**: No master weights needed — persistent VRAM overhead is only the fp32 shift buffer (+2 bytes/param, 0 if offloaded to CPU).
3. **Fixes Kahan**: fp32 shift buffer can actually accumulate sub-threshold compensation across steps.
4. **Is backward-compatible**: All existing optimizer configs work unchanged. Checkpoint loading handles fp32 shift naturally (old checkpoints had bf16 shift → new code creates fp32 from scratch).
5. **Has low risk**: Touches only the update application path; momentum states, muon/adamuon/normuon, automagic LR, and all other logic is unchanged.
