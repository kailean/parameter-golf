# Systematic Issue Analysis — Parameter Golf v3 Warmdown Regression

## Issues Found

### 1. CRITICAL: Warmdown Regression with WD=0.15
**Problem:** val_bpb improved to 1.1684 at step 3000, then REGRESSED to 1.1861 at step 4000.
**Root cause:** WD=0.15 is too aggressive during the LR warmdown phase (steps 3240-4500).
- During warmdown, LR decreases linearly from 1.0 to 0.0
- WD=0.15 keeps pushing weights toward zero
- At low LR, the model can't fight back against weight decay
- Result: weights shrink → model "forgets" → val_bpb gets worse

**Fix options:**
- A) **Reduce WD during warmdown** — Start WD=0.15, decay to 0.095 during warmdown
- B) **Shorter warmdown** — Use 0.50 instead of 0.72 (less time degrading)
- C) **Use EMA checkpoint** — The EMA at step 3000 captures the best weights
- D) **WD schedule** — Apply WD only to non-embedding parameters during warmdown

**Recommended: Option A (WD warmdown schedule)**
```python
# In lr_mul function, also compute WD multiplier:
def wd_mul(step, warmdown_start):
    if step < warmdown_start:
        return 1.0  # Full WD during training
    progress = (step - warmdown_start) / (iterations - warmdown_start)
    return max(1.0 - progress * 0.37, 0.633)  # Decay from 1.0 to 0.633
    # This means WD goes from 0.15 → 0.095 during warmdown
```

### 2. EMA Should Save Best Checkpoint
**Problem:** Current code saves EMA at the END of training, which includes the regressed weights.
**Fix:** Save EMA checkpoint at the step with best val_bpb, not just the last step.

### 3. Compression Size Uncertainty
**Problem:** v1 (512d WD=0.095) compressed to 21.2MB. We expect v3 (512d WD=0.15) to compress better but don't know the exact size.
**Expected:** ~13-14MB at 0.40-0.42 bytes/param ratio (based on v2 results)

### 4. Training Speed on 1×H100
**Problem:** 4500 steps × 1.28s = ~96 min. SOTA trains on 8×H100 in 600s.
**Impact:** More steps = better convergence AND better compression (lower entropy weights)
**Need:** 8×H100 access for proper competition run

### 5. TTT Eval Speed
**Problem:** TTT with LoRA rank 96 on 1×H100 takes 1.5+ hours for sliding eval.
**Fix:** Need 8×H100 for TTT eval within the 600s competition time limit.

## Priority Fixes (for next run)

### Fix 1: WD Warmdown Schedule
```python
# Add to Hyperparameters:
warmdown_wd_decay = float(os.environ.get("WARMDOWN_WD_DECAY", 0.37))  # WD decays by this fraction during warmdown

# In the training loop, when in warmdown:
if step >= warmdown_start:
    wd_progress = (step - warmdown_start) / (iterations - warmdown_start)
    wd_factor = max(1.0 - wd_progress * args.warmdown_wd_decay, 0.0)
    # Apply wd_factor to all WD values
```

### Fix 2: Best Checkpoint Saving
```python
# Track best val_bpb and save EMA when it improves
if val_bpb < best_val_bpb:
    best_val_bpb = val_bpb
    torch.save(ema_state_dict, f"best_model.pt")
```

### Fix 3: Shorter Warmdown (0.50)
```python
WARMDOWN_FRAC=0.50  # Less time degrading = less regression
```

### Fix 4: Iterative WD Reduction
```python
# Start WD=0.15, reduce to 0.095 over warmdown period
# This gives compression benefit during main training
# and quality benefit during warmdown
```

## Recommended Next Config (v4)

```
DIM=512, LAYERS=13, MLP_MULT=4, NUM_HEADS=8, NUM_KV_HEADS=4
MUON_WEIGHT_DECAY=0.15        # Start high for compression
WARMDOWN_WD_DECAY=0.37         # Decay to 0.095 during warmdown
WARMDOWN_FRAC=0.50             # Shorter warmdown (was 0.72)
EMBED_WEIGHT_DECAY=0.15        # Same schedule
ADAM_WEIGHT_DECAY=0.10          # Same schedule
TTT_RANK=96, TTT_STEPS=3       # SOTA TTT
ITERATIONS=4500                 # Full convergence
EMBED_BITS=8                     # Mixed int8/int6
```

This should give us:
- Best BPB from high WD early training
- No warmdown regression from WD schedule
- Better compression from overall high WD
- ~13MB compressed size (fits 16MB!)