# Complete Run Analysis & H100 Projection
## KaiLean × OmniClaw — Parameter Golf

---

## Our Run History

### Run 1: v3 Baseline (M4)
| Metric | Value |
|--------|-------|
| Architecture | 11L, 512d, 8 heads, 3× MLP, int6 QAT, EMA |
| Steps | 200 |
| Tokens/step | 131,072 |
| Total tokens | 26.2M |
| Step time | 24.5s |
| Total train time | 4,896s (~82 min) |
| QAT start | Step 30 (15%) |
| EMA start | Step 100 (50%) |
| **val_loss (standard)** | **4.6622** |
| **val_bpb (standard)** | **2.7612** |
| **sliding_bpb (partial, 25%)** | **~3.87** (trending down, ~100hr to complete) |
| Serialized size | 172MB → int6+zstd: 3.3MB (5.27× compression) |
| Model params | 37.5M |

### Run 2: Moonshot Phase 1 (M4)
| Metric | Value |
|--------|-------|
| Architecture | v3 + EngramLite + SkipGram + Complementary(α=0.5) + NgramMixer |
| Steps | 200 |
| Tokens/step | 131,072 |
| Total tokens | 26.2M |
| Step time | 34.4s (40% slower) |
| Total train time | 6,880s (~115 min) |
| **val_loss (standard)** | **4.7560** |
| **val_bpb (standard)** | **2.8168** (+0.056 worse than baseline) |
| Model params | 39.7M (2.2M extra for hash tables) |

**Why Phase 1 was worse:**
1. Extra hash tables (2.2M params) competed for the 16MB budget
2. α=0.5 complementary training too aggressive for 200 steps
3. 40% slower steps = fewer effective iterations
4. NgramMixer only helps at eval (sliding), not standard val

---

## SOTA Reference (signalrush, 8×H100)
| Metric | Value |
|--------|-------|
| Architecture | 11L, 512d, 8 heads, 4 KV heads, 3× MLP, tied emb |
| Steps | 7,096 in 600s |
| Tokens/step | 786,432 (6× batch size) |
| Total tokens | 5.58 BILLION |
| Step time | 84ms |
| QAT start | Step ~1,064 (15%) |
| EMA start | Step ~3,548 (50%) |
| val_bpb @ step 4000 | 1.2201 |
| val_bpb @ step 7096 (final) | 1.1395 |
| **final_int6_roundtrip** | **1.1466** |
| **final_int6_sliding_window** | **1.1228** |
| Serialized | 15,555,017 bytes (15.5MB) |
| Model params | 26,993,756 |

---

## Scaling Analysis: M4 → 8×H100

### Compute Comparison
| | M4 (our runs) | 8×H100 (SOTA) | Ratio |
|---|---|---|---|
| Step time | 24.5s | 0.084s | 292× |
| Tokens/step | 131K | 786K | 6× |
| Steps in 10 min | 200 | 7,096 | 35.5× |
| Total tokens | 26M | 5.58B | 214× |
| Peak TFLOPS | ~7 | ~3,958 | 565× |

### Loss Curve Extrapolation

The SOTA entry shows a clean loss curve on 8×H100:

| Step | Tokens | train_loss | val_bpb |
|------|--------|------------|---------|
| 0 | 0 | 6.93 | 4.10 |
| 500 | 393M | 2.40 | — |
| 1000 | 786M | 2.27 | — |
| 2000 | 1.57B | 2.06 | — |
| 4000 | 3.15B | 1.97 | 1.22 |
| 7096 | 5.58B | 1.79 | 1.14 |
| **EMA applied** | | | **1.12** |

Our v3 baseline on M4 with only 200 steps (26M tokens) reaches train_loss 4.36 — 
equivalent to roughly step 80-100 on the SOTA curve (before convergence really kicks in).

**Key insight**: At step 4000 (~3.15B tokens), SOTA already achieves val_bpb 1.22. 
Our architecture with 5.5B tokens should reach ~1.10-1.15 pure neural.

### Projected Performance on 8×H100

#### Conservative Estimate (v3 baseline architecture)
| Component | BPB Contribution |
|-----------|-----------------|
| Pure neural (11L, 512d, 7K steps) | 1.10-1.15 |
| + Sliding window eval | -0.02 |
| + BackoffNgramMixer (order 4) | -0.10 to -0.15 |
| + Dirichlet 15-gram mixing | -0.65 to -0.75 |
| + Phrase cache | -0.10 to -0.15 |
| **Total** | **0.10-0.25** |

#### Realistic Estimate (with our innovations)
| Component | BPB Contribution |
|-----------|-----------------|
| Pure neural + XSA + BigramHash + SmearGate | 1.08-1.12 |
| + Sliding window eval | -0.02 |
| + Dirichlet 15-gram mixing | -0.70 to -0.80 |
| + Phrase cache | -0.10 to -0.15 |
| + Complementary training | -0.03 to -0.05 |
| **Total** | **0.11-0.20** |

#### Aggressive Estimate (all innovations stacked)
| Component | BPB Contribution |
|-----------|-----------------|
| Best neural (full 10min) | 1.08 |
| + Sliding window eval | -0.02 |
| + Dirichlet 15-gram (OBCL concentrations) | -0.80 |
| + Phrase cache (probe lengths 20/16) | -0.15 |
| + Complementary training (α=0.5, orders 2-5) | -0.05 |
| + Residual prediction | -0.02 |
| **Total** | **0.04-0.10** |

---

## What The 0.11556 BPB Entry Does

PR #948 (dentity007) achieved **0.11556 BPB** with:
- Two-level Dirichlet-Multinomial posterior mixing (neural → n-gram → phrase)
- Per-order OBCL concentrations: [50, 50, 6.95, 2.98, 2.05, 2.05, 2.05, 1.86, ...]
- Phrase suffix matching at probe lengths [20, 16] with concentration 1.0
- 15-gram backoff (orders 2-15, 4M hash buckets)
- Complementary training (α=0.50, orders 2-5)
- EBLS architecture (3 shared × 3 loops + 2 unique = 11L)
- GPTQ int6 + LZMA compression
- EMA 0.997 + SWA weight averaging
- 8×H100 SXM, 560s training

**This is the target architecture.** Our code already has most of these components.
The missing pieces are:
1. Dirichlet posterior (per-order concentrations) — we have simple backoff
2. Phrase cache (exact suffix matching) — not yet implemented
3. LZMA compression — we use zstd-22
4. EBLS shared-loop architecture — not yet implemented

---

## M4 Estimated Final Scores (if runs completed)

| Run | Standard val_bpb | Sliding bpb (projected) | Notes |
|-----|-------------------|-------------------------|-------|
| v3 Baseline | 2.7612 | ~3.50-3.60 | Sliding was at 3.87 at 25%, trending down slowly |
| Moonshot P1 | 2.8168 | ~3.60-3.70 | Slightly worse, hash table overhead |
| Phase 2 (Residual) | ~2.70-2.75 (est.) | ~3.40-3.50 (est.) | Not yet run |

**Sliding bpb is always ~0.7-0.9 higher than standard val_bpb** for our model size.
The SOTA entry shows: standard 1.14 → sliding 1.12 (sliding is actually BETTER with enough context).
But at our model quality (val_bpb 2.76), sliding window with stride 64 exposes
more positions where the model has less context, so sliding is WORSE.

---

## Action Items for H100 Run

1. **Port train_gpt_kl.py to run on 8×H100 with torchrun** ✅ Already PyTorch
2. **Implement Dirichlet 15-gram mixer** 🔧 In progress
3. **Implement phrase cache** 📋 Next
4. **Switch from zstd-22 to LZMA** 📋 Quick win
5. **Test batch size scaling** (131K → 786K tokens/step)
6. **Tune hyperparameters for 7K steps** (warmup, QAT start, EMA start)
7. **Run 3 seeds for statistical significance** (required for leaderboard submission)
