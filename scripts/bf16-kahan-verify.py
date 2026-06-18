#!/usr/bin/env python3
"""
Numerical verification of Kahan summation variants under bfloat16 emulation.

Tests three variants:
  A. Original "opposite-sign" Kahan (shift.add_(update) pattern) — the alleged bug
  B. Neumaier compensated summation — the proposed replacement
  C. Naive accumulation (no compensation) — baseline

Key question: Does the original variant (A) produce identical compensation as
variant B? Does variant A's compensation actually stay non-zero?

bfloat16 has 7-bit mantissa (vs 10-bit for fp16, 23-bit for fp32).
We emulate by rounding to bf16 after each operation.
"""

import math
import struct
import random

random.seed(42)

# ============================================================
# bfloat16 emulation
# ============================================================

def to_bf16(x):
    """Truncate a Python float to bfloat16 precision."""
    if x == 0.0:
        return 0.0
    # Pack as float32, zero out lower 16 bits of mantissa
    # bf16 = top 16 bits of float32 (1 sign + 8 exp + 7 mantissa)
    bits = struct.unpack('>I', struct.pack('>f', float(x)))[0]
    bits = bits & 0xFFFF0000  # Zero lower 16 bits
    return struct.unpack('>f', struct.pack('>I', bits))[0]


def bf16_add(a, b):
    return to_bf16(a + b)


def bf16_sub(a, b):
    return to_bf16(a - b)


# ============================================================
# Variant A: Original "opposite-sign" Kahan (the alleged bug)
# ============================================================
def kahan_variant_A(updates, p_init=1.0):
    """
    Original code from df7f442:
        shift.add_(update)
        p.grad.copy_(p.detach())    # grad used as temp
        p.add_(shift)
        shift.add_(p.grad.sub_(p))  # grad_old - p_new
    
    With bf16 rounding after each op.
    """
    p = float(p_init)
    shift = 0.0
    history = []
    
    for step, update in enumerate(updates):
        # shift.add_(update)
        shift = bf16_add(shift, update)
        
        # p.grad.copy_(p.detach()) — p_old saved as grad
        p_old = p
        
        # p.add_(shift)
        p = bf16_add(p, shift)
        
        # shift.add_(p.grad.sub_(p))
        # = shift + (p_old - p_new), but with bf16 rounding
        diff = bf16_sub(p_old, p)       # p.grad.sub_(p)
        shift = bf16_add(shift, diff)   # shift.add_(...)
        
        history.append({
            'step': step,
            'update': update,
            'p': p,
            'shift': shift,
            'compensation': shift,
        })
    
    return p, shift, history


# ============================================================
# Variant B: Neumaier compensated summation
# ============================================================
def kahan_variant_B(updates, p_init=1.0):
    """
    Neumaier variant (current code):
        y = update - shift
        t = p + y
        new_shift = (t - p) - y
        p = t
        shift = new_shift
    
    With bf16 rounding after each op.
    """
    p = float(p_init)
    shift = 0.0
    history = []
    
    for step, update in enumerate(updates):
        # y = update - shift
        y = bf16_sub(update, shift)
        
        # t = p + y
        t = bf16_add(p, y)
        
        # new_shift = (t - p) - y
        t_minus_p = bf16_sub(t, p)
        new_shift = bf16_sub(t_minus_p, y)
        
        # p = t
        p = t
        
        # shift = new_shift
        shift = new_shift
        
        history.append({
            'step': step,
            'update': update,
            'p': p,
            'shift': shift,
            'compensation': shift,
        })
    
    return p, shift, history


# ============================================================
# Variant C: Naive accumulation (no compensation)
# ============================================================
def naive_accumulation(updates, p_init=1.0):
    p = float(p_init)
    history = []
    
    for step, update in enumerate(updates):
        p = bf16_add(p, update)
        history.append({
            'step': step,
            'update': update,
            'p': p,
            'shift': 0.0,
        })
    
    return p, 0.0, history


# ============================================================
# Reference: fp32 exact
# ============================================================
def fp32_accumulation(updates, p_init=1.0):
    p = float(p_init)
    for update in updates:
        p += update
    return p


# ============================================================
# Test scenarios
# ============================================================

def scenario_small_updates():
    """Many small updates where compensation matters."""
    return [0.001] * 100  # 100 updates of 0.001

def scenario_mixed_signs():
    """Alternating positive/negative updates of varying magnitude."""
    updates = []
    for i in range(200):
        sign = 1 if i % 2 == 0 else -1
        mag = 10.0 ** random.uniform(-2, 2)  # 0.01 to 100
        updates.append(sign * mag)
    return updates

def scenario_decreasing_magnitudes():
    """Updates that decrease in magnitude (typical training)."""
    updates = [1.0 / (i + 1) for i in range(200)]
    return updates

def scenario_random_walk():
    """Random walk with small increments."""
    updates = [random.uniform(-0.01, 0.01) for _ in range(500)]
    return updates

def scenario_large_cancellation():
    """Large positive then large negative — tests compensation limits."""
    updates = [1000.0, -1000.0, 0.001, 0.001, 0.001, 0.001, 0.001, 
               0.001, 0.001, 0.001, 0.001, 0.001]
    return updates


def run_test(name, updates, p_init=1.0):
    print(f"\n{'='*70}")
    print(f"Test: {name} ({len(updates)} updates)")
    print(f"{'='*70}")
    
    p_naive, _, _ = naive_accumulation(updates, p_init)
    p_A, shift_A, hist_A = kahan_variant_A(updates, p_init)
    p_B, shift_B, hist_B = kahan_variant_B(updates, p_init)
    p_fp32 = fp32_accumulation(updates, p_init)
    
    # Error vs fp32
    err_naive = abs(p_naive - p_fp32)
    err_A = abs(p_A - p_fp32)
    err_B = abs(p_B - p_fp32)
    
    print(f"  fp32 reference:     {p_fp32:.15f}")
    print(f"  Naive bf16:         {p_naive:.15f}  (error: {err_naive:.2e})")
    print(f"  Variant A (orig):   {p_A:.15f}  (error: {err_A:.2e})")
    print(f"  Variant B (Neum):   {p_B:.15f}  (error: {err_B:.2e})")
    print(f"")
    print(f"  Final shift A:      {shift_A:.15f}")
    print(f"  Final shift B:      {shift_B:.15f}")
    print(f"  A ≡ B?              {p_A == p_B and shift_A == shift_B}")
    print(f"  A better than naive? {err_A < err_naive}")
    print(f"  B better than naive? {err_B < err_naive}")
    
    # Track compensation non-zero count
    nonzero_A = sum(1 for h in hist_A if abs(h['shift']) > 1e-30)
    nonzero_B = sum(1 for h in hist_B if abs(h['shift']) > 1e-30)
    print(f"  Steps with non-zero compensation A: {nonzero_A}/{len(hist_A)}")
    print(f"  Steps with non-zero compensation B: {nonzero_B}/{len(hist_B)}")
    
    # Check if A and B are bit-identical at each step
    identical_steps = 0
    for ha, hb in zip(hist_A, hist_B):
        if ha['p'] == hb['p'] and ha['shift'] == hb['shift']:
            identical_steps += 1
    print(f"  Bit-identical steps: {identical_steps}/{len(hist_A)}")
    
    # Track compensation magnitude over time
    comp_mag_A = [abs(h['shift']) for h in hist_A]
    comp_mag_B = [abs(h['shift']) for h in hist_B]
    if max(comp_mag_A) > 1e-10:
        print(f"  Max |shift| A:      {max(comp_mag_A):.6e}")
    if max(comp_mag_B) > 1e-10:
        print(f"  Max |shift| B:      {max(comp_mag_B):.6e}")
    
    return {
        'name': name,
        'p_fp32': p_fp32,
        'p_A': p_A, 'p_B': p_B, 'p_naive': p_naive,
        'err_A': err_A, 'err_B': err_B, 'err_naive': err_naive,
        'shift_A': shift_A, 'shift_B': shift_B,
        'A_better_than_naive': err_A < err_naive,
        'B_better_than_naive': err_B < err_naive,
        'nonzero_A': nonzero_A,
        'nonzero_B': nonzero_B,
        'identical_steps': identical_steps,
        'total_steps': len(hist_A),
    }


if __name__ == '__main__':
    results = []
    
    # Scenario 1: Small identical updates
    results.append(run_test("Small identical updates (+0.001 × 100)", scenario_small_updates()))
    
    # Scenario 2: Mixed sign, varying magnitude
    results.append(run_test("Mixed signs, varying magnitude", scenario_mixed_signs()))
    
    # Scenario 3: Decreasing magnitudes
    results.append(run_test("Decreasing magnitudes", scenario_decreasing_magnitudes()))
    
    # Scenario 4: Random walk small increments
    results.append(run_test("Random walk (±0.01 × 500)", scenario_random_walk()))
    
    # Scenario 5: Large cancellation
    results.append(run_test("Large cancellation (+1000, -1000, +tiny...)", scenario_large_cancellation()))
    
    # Scenario 6: Realistic training — gradient updates with noise
    # Simulate noisy gradients around a trend
    random.seed(123)
    updates = []
    trend = 0.0
    for i in range(1000):
        trend = 0.99 * trend + 0.01 * 0.001  # decaying trend toward +0.001
        noise = random.gauss(0, 0.01)
        updates.append(trend + noise)
    results.append(run_test("Realistic training (noisy grad × 1000)", updates))
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    total_A_err = sum(r['err_A'] for r in results)
    total_B_err = sum(r['err_B'] for r in results)
    total_naive_err = sum(r['err_naive'] for r in results)
    
    print(f"  Total error (all tests):")
    print(f"    Naive:    {total_naive_err:.2e}")
    print(f"    Variant A: {total_A_err:.2e}")
    print(f"    Variant B: {total_B_err:.2e}")
    
    all_identical = all(r['identical_steps'] == r['total_steps'] for r in results)
    print(f"\n  A and B bit-identical in all tests: {all_identical}")
    
    # Key question: does variant A's compensation actually work?
    any_nonzero_A = any(r['nonzero_A'] > 0 for r in results)
    print(f"  Variant A has non-zero compensation in any test: {any_nonzero_A}")
    
    any_better_A = any(r['A_better_than_naive'] for r in results)
    any_better_B = any(r['B_better_than_naive'] for r in results)
    print(f"  Variant A beat naive in any test: {any_better_A}")
    print(f"  Variant B beat naive in any test: {any_better_B}")
