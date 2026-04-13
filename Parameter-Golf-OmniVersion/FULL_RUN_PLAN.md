# Parameter Golf — Full Competition Run Plan
## KaiLean × OmniClaw | April 10, 2026

---

## Smoke Test Results (Dirichlet Mixer)

| Metric | Neural Only | + Dirichlet Mixer | Delta |
|--------|------------|-------------------|-------|
| Standard val_bpb | 2.7610 | — | — |
| Sliding val_bpb | ~3.90 | ~2.63 | **-1.27 (32.5%)** |
| Model size (int6+zstd) | 3.3MB | 3.3MB | — |
| Params | 43.3M | 43.3M | — |

**Key insight:** The Dirichlet n-gram mixer alone provides a 1.27 BPB improvement.
The neural model's role is to provide residual prediction for what n-grams can't capture.

---

## Competition Landscape

| Rank | BPB | Technique |
|------|-----|-----------|
| 1 | 0.1156 | Dirichlet 15-gram + phrase cache + EBLS (dentity007) |
| 3 | 1.0226 | Gated DeltaNet (buggy eval) |
| 4 | 1.1086 | signalrush: SOTA clean entry |
| **Us** | **~2.63** | **Dirichlet mixer + 200-step smoke test** |

**Gap to close:** ~1.53 BPB to reach #4 (signalrush), ~2.52 BPB to reach #1.

---

## Phase Analysis: Where the BPB Will Come From

### Phase 1: Full Training Run (expected: -1.0 to -1.5 BPB)
Current smoke test was 200 steps on M4. A proper run needs:
- **10,000-20,000 iterations** (vs 200)
- **Full FineWeb 10B training data** (we used all 10 shards but only 200 steps)
- **Proper warmup/warmdown scheduling**
- **GPU training** (H100) for speed

The current 2.63 BPB is with a severely undertrained model (200 steps).
A fully trained model should reach ~1.1-1.5 BPB neural-only, which
with Dirichlet mixing should land at ~0.8-1.3 BPB.

### Phase 2: Residual Prediction Architecture (expected: -0.05 to -0.15 BPB)
- Train model to predict ONLY the residual (what n-grams can't capture)
- n-gram logits added BEFORE softmax, not mixed after
- Code already exists: `residual_loss()` in train_gpt_mlx_kl.py
- Needs `RESIDUAL_PREDICTION=1` and `COMPLEMENT_ALPHA=0.001`

### Phase 3: Compression-Aware Training (expected: -0.003 to -0.01 BPB)
- Differentiable entropy penalty on int6-quantized weights
- Code already exists: `compress_aware_loss()` 
- Needs `COMPRESS_AWARE=1`
- Small gain but free

### Phase 4: Byte-Weighted Loss (expected: -0.003 to -0.01 BPB)
- Weight CE loss by bytes-per-token to directly optimize BPB
- Code already exists: `byte_weighted_loss()`
- Needs `BYTE_WEIGHTED_LOSS=1`

### Phase 5: Optimized Dirichlet Mixer (expected: -0.05 to -0.2 BPB)
- Tune concentrations per-order (currently using OBCL defaults)
- Optimize phrase cache probe lengths
- Try higher hash buckets (4M → 8M)
- Adjust mixing alpha range

### Phase 6: GPU Port + Full Training (required for competition)
- Port training to PyTorch/CUDA for 8×H100
- 10-minute training limit means ~730 steps at 820ms/step on H100
- Or: train on M4 overnight (no time limit), submit quantized model

---

## Action Plan

### Step 1: Overnight M4 Full Training Run (START NOW)
Run on M4 with all innovations enabled. No GPU needed.

```bash
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export MLP_MULT=3
export VOCAB_SIZE=1024
export ITERATIONS=10000
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=2000
export TRAIN_BATCH_TOKENS=524288
export GRAD_ACCUM_STEPS=8
export TRAIN_SEQ_LEN=1024
export SEED=1337
export QAT_START_FRAC=0.15
export EMA_DECAY=0.995
export EMA_START_FRAC=0.5
export DIRICHLET_MIXER=1
export DIRICHLET_MAX_ORDER=15
export DIRICHLET_ALPHA=0.25
export PHRASE_CACHE_ENABLED=1
export PHRASE_PROBE_LENGTHS="20,16"
export NGRAM_MIXER_ENABLED=1
export RESIDUAL_PREDICTION=1
export COMPLEMENT_ALPHA=0.001
export COMPRESS_AWARE=1
export COMPRESS_AWARE_WEIGHT=0.01
export BYTE_WEIGHTED_LOSS=1
export EVAL_MODE=both
export OUT_DIR="logs/full_run_v1"
export RUN_ID="full_v1"
```

Estimated time on M4: ~6-8 hours for 10K iterations
This gets us a properly trained model + Dirichlet eval.

### Step 2: Check Compute Grant Status
- Applied April 9 for $1000 / ~320 H100 hours
- Should hear back within 1-2 business days
- If granted: port to PyTorch/CUDA for competition submission

### Step 3: Hyperparameter Sweep (if time permits)
- Try different model sizes (512d vs 768d)
- Try different Dirichlet concentrations
- Try different phrase probe lengths
- Try different QAT thresholds

### Step 4: Competition Submission
- Must fit in 16MB (currently 3.3MB — plenty of room)
- 10 min training on 8×H100
- 10 min eval
- Submit compressed model + eval script

---

## Risk Assessment

1. **M4 training is slow** — 10K iterations could take 6-8 hours. Plan for overnight runs.
2. **Dirichlet eval is very slow on CPU** — 969K windows took 12+ hours. Need GPU eval.
3. **Compute grant uncertainty** — if denied, we need an alternative GPU source.
4. **Model size is fine** — 3.3MB compressed, well under 16MB budget.
5. **Competition rules** — must verify our approach complies with all submission rules.

---

## Expected Final BPB Estimate

| Phase | Technique | Expected BPB |
|-------|-----------|-------------|
| Current | Smoke test (200 steps) + Dirichlet | ~2.63 |
| + Full training | 10K steps, proper schedule | ~1.0-1.3 |
| + Residual prediction | Train on residual only | ~0.9-1.2 |
| + Compression-aware | Entropy regularization | ~0.9-1.2 |
| + Byte-weighted loss | BPB-optimized training | ~0.89-1.19 |
| + Dirichlet tuning | Optimize concentrations | ~0.85-1.15 |
| **Target** | **Competition submission** | **~0.85-1.15** |

This would put us competitive with #4 (signalrush at 1.1086) and
potentially in top-3 territory. Reaching #1 (0.1156) would require
the EBLS technique which we haven't implemented yet.