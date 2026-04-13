# Parameter Golf: Deep Analysis & Innovation Roadmap
## KaiLean × OmniClaw — April 9, 2026

## The Brutal Truth About Our Position

Our v3 baseline: **2.7612 BPB** (standard val)
Competition SOTA leaderboard: **1.1228 BPB** (signalrush)
Actual best submissions: **0.11556 BPB** (Dirichlet n-gram + phrase cache)
The gap from us to SOTA: **1.64 BPB** (we're 2.5× worse)
The gap from leaderboard SOTA to best n-gram: **1.01 BPB** (n-gram mixer alone)

## The Three Leagues of This Competition

### League 1: Pure Neural (1.10–1.22 BPB)
The "official" leaderboard entries. No external data, no eval-time caches, no n-gram mixing.
- SOTA: 1.0226 (Gated DeltaNet, buggy but conceptually proven)
- Clean SOTA: 1.1086 (signalrush)
- Our v3 baseline: 2.7612 (way off)

**Key insight**: The pure neural SOTA is ~1.10 BPB. Our 2.76 is not even in the same league.

### League 2: Neural + N-gram Cache (0.40–1.10 BPB)
Post-normalization n-gram mixing. These add a BackoffNgramMixer at eval time.
- Causal n-gram mixer with proper normalization: ~0.40-0.44 BPB standalone
- When mixed with a 1.10 BPB neural model, achieves sub-1.0

### League 3: Dirichlet + Phrase Cache (0.11-0.20 BPB)
The current frontier. Two-level Dirichlet posterior + per-order concentration tuning + phrase matching.
- PR #948: 0.11556 BPB with 15-gram backoff + Dirichlet smoothing
- These are basically n-gram models on steroids

## Why We're at 2.76 (Not 1.10)

1. **We train on M4, not 8×H100**: ~30× slower. The competition runs 600 seconds on 8×H100 = ~540K tokens/step × 200 steps × 8 GPUs ≈ 864M tokens. We see ~26M tokens × 200 = 5.2B tokens. They see 10-100× more data.

2. **200 steps is nothing**: The baseline was trained to step 13,780 (20K steps). We train 200 steps. The competition uses 10 min on 8×H100 = thousands of effective steps.

3. **Our architecture is solid but undertrained**: The innovations (BigramHash, SmearGate, XSA, EMA, int6 QAT) are exactly what the top entries use. The problem is compute, not architecture.

## The Missing Pieces (In Priority Order)

### 1. COMPUTE: Get 8×H100 Access
This is THE bottleneck. Our M4 training gives us 200 steps × 131K tokens = 26M tokens/step. The competition uses 8×H100 for 10 minutes.

Options:
- Apply for OpenAI compute grant ($1M in credits available)
- Use Runpod/GPU rental (~$15-20/hr for 8×H100)
- Our current `train_gpt_kl.py` is PyTorch-ready for CUDA

**Expected improvement: 2.76 → ~1.10-1.15 BPB** (just from proper training)

### 2. N-gram Cache (The Biggest Eval-Time Win)
We already have BackoffNgramMixer in our code. It works. But:

**What we have**: Orders 2-4, hash buckets, entropy-adaptive alpha
**What the top entries use**:
- Orders 2-15 (yes, 15-gram!)
- Two-level Dirichlet posterior smoothing
- Per-order concentration tuning (not just simple backoff)
- Phrase cache (exact suffix matching at probe lengths 20/16)
- OBCL (Order-By-Conditional-Log-likelihood) for concentration estimation

**Expected improvement: ~0.15-0.40 BPB** (on top of neural model)

### 3. Dirichlet Posterior Smoothing
Simple backoff (Katz/interpolated) gives ~0.40 BPB standalone.
Dirichlet-Multinomial posterior with per-order concentrations gives ~0.12 BPB standalone.

The difference: instead of `P(word|context) = count(word,context) / count(context)`,
use `P(word|context) = (count(word,context) + alpha * prior(word)) / (count(context) + alpha)`,
where alpha is tuned per order: higher for low-order (smoother), lower for high-order (more specific).

### 4. Gated DeltaNet Architecture
The 1.0226 BPB entry used GatedDeltaNet layers instead of standard attention.
Key insight: recurrence across chunks gives unlimited context without quadratic cost.
Our standard attention is limited to seq_len=1024 context.

### 5. LZMA/Brotli Compression (Not zstd-22)
We use zstd-22. PR #948 uses LZMA (xz). The competition allows any compression.
LZMA achieves better compression ratios at the cost of speed (which doesn't matter for 16MB).

## The Unasked Questions

### Q1: Why are we using MLX instead of CUDA?
MLX is great for Apple Silicon development but 30× slower than 8×H100.
We have a PyTorch version (`train_gpt_kl.py`) ready. We should be running on GPU.

### Q2: Why only 200 steps?
We set ITERATIONS=200 for fast local iteration. The competition allows 600 seconds on 8×H100.
On GPU, that's thousands of steps. We should match the competition budget.

### Q3: Why aren't we using higher-order n-gram mixing?
Our BackoffNgramMixer only goes to order 4. The SOTA uses orders 2-15.
The Dirichlet posterior + per-order concentrations is the key differentiator.

### Q4: Why aren't we compressing with LZMA instead of zstd?
LZMA/xz achieves better ratios for model weights. The competition allows it.

### Q5: What about Gated DeltaNet?
The 1.0226 entry proved recurrence helps. We're using standard attention.
Replacing some attention layers with DeltaNet could give 0.05-0.10 BPB.

## Concrete Action Plan

### Phase A: GPU Migration (Immediate, ~2 hours)
1. Set up Runpod/8×H100
2. Transfer `train_gpt_kl.py` (PyTorch version)
3. Run v3 baseline on 8×H100 with 600s wallclock
4. Target: ~1.10-1.15 BPB

### Phase B: Dirichlet N-gram Cache (Day 2, ~4 hours)
1. Extend BackoffNgramMixer to orders 2-15
2. Implement Dirichlet posterior with per-order concentrations
3. Add phrase cache (exact suffix matching)
4. Target: ~0.80-0.95 BPB combined

### Phase C: Architecture Upgrades (Day 3-4)
1. Replace last 2-3 attention layers with GatedDeltaNet
2. Add complementary training (alpha=0.50, orders 2-5)
3. Switch compression from zstd-22 to LZMA
4. Target: ~0.60-0.80 BPB

### Phase D: Full Optimization (Day 5+)
1. Per-order concentration tuning (grid search)
2. Phrase cache probe length optimization
3. EBLS (shared-loop) architecture
4. GPTQ calibration for int6
5. Target: <0.50 BPB

## The Math

Pure neural SOTA: ~1.10 BPB
+ Dirichlet 15-gram cache: ~-0.70 BPB
+ Phrase cache: ~-0.15 BPB
+ Complementary training: ~-0.05 BPB
+ LZMA compression: ~-0.01 BPB
= **Target: ~0.19 BPB**

Even being conservative: 1.10 × 0.3 (n-gram dominates) + overhead ≈ **0.33-0.50 BPB**

This is achievable. The 0.11556 entry proves it.

## What I Need From You

1. **8×H100 access** — Apply for OpenAI compute grant or set up Runpod
2. **Permission to port train_gpt_kl.py to CUDA** — Already have the PyTorch version
3. **Time budget** — Each full run takes 10 minutes on 8×H100

The M4 runs are useful for development and debugging, but we cannot compete on this hardware. The compute gap is the #1 blocker.