#!/usr/bin/env python3
"""Diagnostic: bf16 Kahan summation precision analysis - pure Python simulation.

Simulates bf16 arithmetic using custom rounding to bfloat16 precision.
bf16: 1 sign, 8 exponent, 7 mantissa = 16 bits total.
"""

import struct
import math


def to_bf16(x):
    """Convert float to bfloat16 (truncate mantissa to 7 bits)."""
    if x == 0.0:
        return 0.0
    if math.isinf(x) or math.isnan(x):
        return x
    # Pack as fp32, get the bit pattern
    bits = struct.unpack('I', struct.pack('f', float(x)))[0]
    # bf16 has 7 mantissa bits (16 total mantissa bits in fp32, keep top 7)
    # Round to nearest even
    rounding_bits = bits & 0xFFFF
    truncated = bits & 0xFFFF0000
    
    # Round to nearest even
    if rounding_bits > 0x8000:
        truncated += 0x10000
    elif rounding_bits == 0x8000:
        # Tie: round to even
        if truncated & 0x10000:
            truncated += 0x10000
    
    # Unpack back
    return struct.unpack('f', struct.pack('I', truncated))[0]


def bf16_add(a, b):
    return to_bf16(a + b)


def bf16_sub(a, b):
    return to_bf16(a - b)


def bf16_mul(a, b):
    return to_bf16(a * b)


def simulate_kahan_current(p_init, update_per_step, num_steps):
    """Simulate the CURRENT Kahan summation in generic_optim.py."""
    p = to_bf16(p_init)
    shift = to_bf16(0.0)
    history = []
    
    for step in range(num_steps):
        update = to_bf16(-update_per_step)  # negated for gradient descent
        
        # Current implementation (lines 490-501):
        # shift.add_(update)
        shift = bf16_add(shift, update)
        # p.grad.copy_(p.detach()) — save p_old
        p_old = p
        # p.add_(shift)
        p_new = bf16_add(p, shift)
        # shift.add_(p.grad.sub_(p)) — p.grad.sub_(p) = p_old - p_new
        recovery = bf16_sub(p_old, p_new)
        shift = bf16_add(shift, recovery)
        p = p_new
        
        effective_change = to_bf16(p - p_old)
        exact_p = to_bf16(p_init - update_per_step * (step + 1))
        
        history.append({
            'step': step,
            'p': p,
            'shift': shift,
            'effective_change': effective_change,
            'exact_p': exact_p,
        })
    
    return history


def simulate_kahan_fix(p_init, update_per_step, num_steps):
    """Simulate the PROPOSED FIX (Neumaier formulation)."""
    p = to_bf16(p_init)
    shift = to_bf16(0.0)
    history = []
    
    for step in range(num_steps):
        update = to_bf16(-update_per_step)
        
        # Fix (Neumaier):
        y = bf16_sub(update, shift)
        t = bf16_add(p, y)
        new_shift = bf16_sub(bf16_sub(t, p), y)
        p = t
        shift = new_shift
        
        exact_p = to_bf16(p_init - update_per_step * (step + 1))
        
        history.append({
            'step': step,
            'p': p,
            'shift': shift,
            'exact_p': exact_p,
        })
    
    return history


def simulate_direct(p_init, update_per_step, num_steps):
    """Simulate direct addition (no Kahan)."""
    p = to_bf16(p_init)
    history = []
    
    for step in range(num_steps):
        p_old = p
        update = to_bf16(-update_per_step)
        p = bf16_add(p, update)
        effective_change = to_bf16(p - p_old)
        exact_p = to_bf16(p_init - update_per_step * (step + 1))
        
        history.append({
            'step': step,
            'p': p,
            'effective_change': effective_change,
            'exact_p': exact_p,
        })
    
    return history


def simulate_kahan_fp32_shift(p_init, update_per_step, num_steps):
    """Simulate Kahan with bf16 parameter but fp32 shift (what the fix ideally should do)."""
    p = to_bf16(p_init)
    shift = float(0.0)  # FP32 precision for shift
    history = []
    
    for step in range(num_steps):
        update = to_bf16(-update_per_step)
        
        # Compute with fp32 shift, apply to bf16 p
        shift = float(shift) + float(update)  # fp32 accumulation
        p_old = p
        p_new = bf16_add(p, to_bf16(shift))  # quantize shift to bf16 for addition
        recovery = bf16_sub(p_old, p_new)     # bf16 recovery
        shift = float(shift) + float(recovery)  # fp32 accumulation of recovery
        
        p = p_new
        effective_change = to_bf16(p - p_old)
        exact_p = to_bf16(p_init - update_per_step * (step + 1))
        
        history.append({
            'step': step,
            'p': p,
            'shift': shift,
            'effective_change': effective_change,
            'exact_p': exact_p,
        })
    
    return history


def analyze(name, p_init, lr, grad_scale, num_steps, sim_fn, **kwargs):
    """Run simulation and print analysis."""
    update_per_step = lr * grad_scale
    bf16_eps_at_p = p_init * 2**-7
    
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  p_init={p_init}, lr={lr}, grad_scale={grad_scale}, update/step={update_per_step}")
    print(f"  bf16 eps at p={p_init}: {bf16_eps_at_p}")
    print(f"  Steps to reach bf16 eps: {bf16_eps_at_p / update_per_step:.1f}")
    print(f"{'='*70}")
    
    history = sim_fn(p_init, update_per_step, num_steps, **kwargs)
    
    # Find actual changes
    changes = []
    prev_p = p_init
    for h in history:
        if h['p'] != prev_p:
            changes.append(h)
        prev_p = h['p']
    
    zero_change_steps = sum(1 for h in history if h.get('effective_change', 
                             h['p'] - history[max(0, h['step']-1)]['p'] if h['step'] > 0 else 0) == 0.0)
    
    if changes:
        print(f"  Parameter changed at steps: {[c['step']+1 for c in changes]}")
        for c in changes[:10]:
            print(f"    Step {c['step']+1:4d}: p={c['p']:12.10f}, shift={c['shift']:16.10e}, "
                  f"exact={c['exact_p']:12.10f}")
        if len(changes) > 10:
            print(f"    ... and {len(changes)-10} more changes")
    else:
        print(f"  Parameter NEVER changed in {num_steps} steps!")
    
    final = history[-1]
    print(f"  Final: p={final['p']:.10f}, exact={final['exact_p']:.10f}, "
          f"abs_err={abs(final['p']-final['exact_p']):.2e}")
    print(f"  Zero-change steps: {zero_change_steps}/{num_steps}")
    
    return history


def check_shift_precision_loss():
    """Check if bf16 shift loses precision for small increments."""
    print(f"\n{'='*70}")
    print(f"  SHIFT PRECISION LOSS CHECK (bf16 shift, 8e-6 increment)")
    print(f"{'='*70}")
    
    shift = to_bf16(0.0)
    update = to_bf16(8e-6)
    for i in range(1, 2001):
        shift_before = shift
        shift = bf16_add(shift, update)
        if shift == shift_before and i > 1:
            print(f"  Shift stopped accumulating at step {i}!")
            print(f"  shift={shift:.10f}, bf16 eps at this magnitude: {shift * 2**-7:.2e}")
            print(f"  update={update:.2e} < eps={shift * 2**-7:.2e} → update lost!")
            return i
    else:
        print(f"  Shift reached {shift:.10f} after 2000 steps (no loss yet)")
    return None


if __name__ == "__main__":
    print("=" * 70)
    print("BF16 KAHAN SUMMATION AUDIT — Pure Python bf16 Simulation")
    print("=" * 70)
    
    # Scenario 1: Standard LR, gradients normalized by Adam (magnitude ~1)
    analyze("SCENARIO 1: Current Kahan (bf16)", 1.0, 8e-6, 1.0, 200, simulate_kahan_current)
    analyze("SCENARIO 1: Neumaier fix (bf16)", 1.0, 8e-6, 1.0, 200, simulate_kahan_fix)
    analyze("SCENARIO 1: Direct (no Kahan)", 1.0, 8e-6, 1.0, 200, simulate_direct)
    analyze("SCENARIO 1: Kahan fp32 shift", 1.0, 8e-6, 1.0, 200, simulate_kahan_fp32_shift)
    
    # Scenario 2: LR=6e-6
    analyze("SCENARIO 2: Current Kahan (bf16)", 1.0, 6e-6, 1.0, 200, simulate_kahan_current)
    analyze("SCENARIO 2: Neumaier fix (bf16)", 1.0, 6e-6, 1.0, 200, simulate_kahan_fix)
    analyze("SCENARIO 2: Kahan fp32 shift", 1.0, 6e-6, 1.0, 200, simulate_kahan_fp32_shift)
    
    # Scenario 3: With bias correction at step ~10 (effective LR reduced ~6x)
    # Ratio = sqrt(1-beta2^step) / (1-beta1^step). At step=10: sqrt(1-0.999^10)/(1-0.9^10) ≈ 0.153
    analyze("SCENARIO 3: W/bias corr (eff lr≈1.2e-6)", 1.0, 8e-6 * 0.153, 1.0, 200, simulate_kahan_current)
    analyze("SCENARIO 3: Kahan fp32 shift", 1.0, 8e-6 * 0.153, 1.0, 200, simulate_kahan_fp32_shift)
    
    # Scenario 4: Very small initial parameter values  
    analyze("SCENARIO 4: Small p_init=0.1, Current Kahan", 0.1, 8e-6, 1.0, 200, simulate_kahan_current)
    analyze("SCENARIO 4: Small p_init=0.1, Kahan fp32 shift", 0.1, 8e-6, 1.0, 200, simulate_kahan_fp32_shift)
    
    # Check shift precision
    check_shift_precision_loss()
    
    # Compute bf16 precision limits at various magnitudes
    print(f"\n{'='*70}")
    print(f"  BF16 PRECISION AT VARIOUS MAGNITUDES")
    print(f"{'='*70}")
    for mag in [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001, 8e-6]:
        eps = mag * 2**-7
        print(f"  |p|={mag:12.10f}: bf16 eps = {eps:16.10e}")
