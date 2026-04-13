# H100 Performance Projection for KaiLean's Parameter Golf Entry

## Hardware Comparison

| Metric | M4 Mac (our runs) | 8×H100 SXM (competition) |
|--------|-------------------|--------------------------|
| GPU | Apple M4 (10-core GPU) | 8× NVIDIA H100 SXM 80GB |
| Memory | 16GB unified | 640GB HBM3 |
| Peak TFLOPS | ~7 (FP16) | ~3,958 (FP16, 8×) |
| Memory BW | ~120 GB/s | ~25,600 GB/s (8×) |
| Our step time | 24,500 ms | ~84 ms |
| Speedup factor | 1× | **~292×** |

## SOTA Entry Analysis (signalrush, 1.1228 BPB)

From `train.log`:
- **Architecture**: 11L, 512d, 8 heads, 4 KV heads, 3× MLP, tied embeddings
- **Training**: 20,000 steps, 786,432 tokens/step, 8×H100
- **Step time**: ~84ms/step on 8×H100
- **Total train time**: 600s (10 min wallclock cap)
- **Steps completed**: 7,096 in 10 min
- **Total tokens seen**: 7,096 × 786,432 = ~5.58 BILLION tokens
- **Final val_bpb**: 1.1228 (sliding window)

## Our Runs vs Competition

| Metric | v3 Baseline (M4) | Moonshot P1 (M4) | SOTA (8×H100) |
|--------|-------------------|-------------------|----------------|
| Steps | 200 | 200 | 7,096 |
| Tokens/step | 131,072 | 131,072 | 786,432 |
| Total tokens | 26.2M | 26.2M | 5.58B |
| Step time | 24.5s | 34.4s | 0.084s |
| val_bpb | 2.7612 | 2.8168 | **1.1228** |

**Our M4 runs see ~214× fewer tokens than a full H100 run.**

## Projected Performance on 8×H100

### Method 1: Scaling from SOTA
The SOTA entry uses essentially the same architecture as our v3 baseline:
- 11L, 512d, 8 heads, 4 KV heads, 3× MLP
- Muon optimizer, EMA, int6 QAT, XSA
- Our additions: BigramHash, SmearGate, BackoffNgramMixer

If we ran our v3 baseline on 8×H100 with 600s:
- We'd see ~7,000 steps (same as SOTA)
- With same batch size (786,432 tokens/step), we'd see ~5.5B tokens
- Our architecture is slightly larger (37.5M vs 27M params) due to BigramHash
- **Expected pure neural BPB: ~1.12-1.15** (slightly worse than SOTA due to larger model)

### Method 2: Scaling from our M4 results
Our v3 baseline at step 200 (26M tokens) = 2.76 BPB
SOTA baseline at step 200 (157M tokens) would be ~2.0 BPB
The gap is mostly data/step count, not architecture.

### Projected Scores on 8×H100

| Configuration | Pure Neural BPB | +BackoffNgramMixer (order 4) | +Dirichlet 15-gram | +Phrase Cache |
|--------------|-----------------|------------------------------|-------------------|---------------|
| v3 Baseline (11L, 512d) | 1.12-1.15 | ~0.95-1.00 | ~0.30-0.40 | ~0.15-0.20 |
| +BigramHash+XSA | 1.10-1.13 | ~0.93-0.98 | ~0.28-0.38 | ~0.14-0.18 |
| +EngramLite+SkipGram | 1.11-1.14 | ~0.94-0.99 | ~0.29-0.39 | ~0.15-0.19 |
| +Residual Prediction | 1.09-1.12 | ~0.92-0.97 | ~0.27-0.37 | ~0.13-0.17 |

### Conservative Estimate
**Pure neural: ~1.11 BPB** (matching SOTA)
**With Dirichlet 15-gram + phrase cache: ~0.15-0.25 BPB**

### Optimistic Estimate
**Pure neural: ~1.08 BPB** (our innovations help on full training)
**With Dirichlet 15-gram + phrase cache: ~0.12-0.18 BPB**

## What the 8×H100 Run Would Look Like

```bash
# Our training script on 8×H100
torchrun --standalone --nproc_per_node=8 train_gpt_kl.py \
  NUM_LAYERS=11 MODEL_DIM=512 NUM_HEADS=8 MLP_MULT=3 \
  ITERATIONS=20000 TRAIN_BATCH_TOKENS=786432 \
  MAX_WALLCLOCK_SECONDS=600 \
  QAT_START_FRAC=0.15 EMA_DECAY=0.995 \
  NGRAM_MIXER_ENABLED=1 NGRAM_ALPHA=0.25 NGRAM_MAX_ORDER=15 \
  EVAL_MODE=sliding
```

Timeline:
- Steps 0-20: Warmup (~1.7s)
- Steps 20-7096: Training (~600s)
- QAT starts at step ~1064 (15% of 7096)
- EMA starts at step ~3548 (50% of 7096)
- Final eval + sliding window + n-gram mixing: ~73s
- Total: ~674s ≈ 11.2 min (slightly over 10 min cap — need to optimize)

**Critical**: We need to fit training + eval in 10 min (600s). The SOTA entry finishes training in 600s and eval in 73s separately. Our script needs the same separation.

## Token Budget Analysis

| Config | Tokens/step | Steps in 600s | Total tokens |
|--------|-------------|----------------|--------------|
| 1×H100, bs=8192 | ~8K | ~7,000 | ~56M |
| 8×H100, bs=786432 | ~786K | ~7,000 | ~5.5B |
| Our M4, bs=131072 | ~131K | 200 | ~26M |

The M4 sees 214× fewer tokens than H100. The scaling from 26M → 5.5B tokens is the primary driver of the 2.76 → 1.12 BPB improvement.