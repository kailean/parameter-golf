# Sub-1.0 BPB Moonshot Strategy
## KaiLean × OmniClaw — April 2026

### Current State
- v3 baseline: val_bpb 2.76 (standard), ~3.87 (sliding) — far from target
- Phase 1 running: EngramLite + SkipGram + Complementary Training + BackoffNgramMixer
- Competition SOTA: ~1.11 BPB (official), ~1.10 (pending)
- **Target: sub-1.0 BPB**

### The 3 Questions Nobody Has Asked

**Q1: Residual Prediction — not complementary loss, but residual architecture**
Instead of training model to predict P(token|context) and then mixing with n-grams,
train it to predict ONLY what n-grams cannot. The model output is added to n-gram
logits, not mixed. This means the model can allocate 100% of its capacity to the
"hard" distribution — the part n-grams can't handle.

**Q2: Compression-Aware Training — optimize for zstd ratio, not just loss**
Instead of train → quantize → compress, add a differentiable proxy for compression
ratio to the training loss. Per-layer byte shuffling permutations that minimize
entropy of the quantized weight stream. The shuffle table costs ~100 bytes but can
save 0.5-1MB of compressed artifact.

**Q3: Adaptive Token Vocabulary — the tokenizer is a hidden variable**
SP1024 is fixed, but what if we learn which tokens to merge/split? Or use a
dual-vocabulary approach where high-frequency n-grams get dedicated bypass tokens
that the n-gram mixer handles without the neural model?

### Architecture: The Full Stack

```
TRAINING:
1. BackoffNgramMixer builds causal n-gram stats from training data (ALPHA)
2. Model produces RESIDUAL logits (not full distribution)
3. Loss = CE(softmax(ngram_logits + residual_logits), targets)
4. This forces the model to specialize entirely on what n-grams miss
5. EngramLite + SkipGram provide n-gram context during training
6. Compression-aware regularization pushes weights toward zstd-friendly distributions

EVAL:
1. Load quantized model (standard)
2. For each token:
   a. Build n-gram prediction P_ngram from BackoffNgramMixer (orders 2-10)
   b. Get model residual logits r
   c. Final: P = softmax(ngram_logits + r)
   d. Score, update n-gram cache
3. Result: complementary predictions from specialized components
```

### Implementation Roadmap

#### Phase 1 (running now): Unlock existing innovations
- EngramLite + SkipGram + Complementary Training + BackoffNgramMixer
- Expected: ~1.0-1.1 BPB over baseline (if convergence is good)

#### Phase 2: Residual Prediction Architecture
- Modify model to output residual logits that ADD to n-gram logits
- This is NOT complementary loss — it's a fundamental change to what the model learns
- The n-gram component handles all local patterns; the residual model handles
  long-range dependencies, rare tokens, and novel combinations
- Expected: 0.05-0.15 BPB over Phase 1

#### Phase 3: Compression-Aware Training
- Add differentiable proxy for zstd compression ratio to training loss
- Learn per-layer byte-shuffle permutations
- NuMuon optimizer for lower stable rank (better compression)
- Expected: 0.003-0.01 BPB

#### Phase 4: Tokenizer & Eval Optimization
- Analyze SP1024 token efficiency (bytes/token distribution)
- If possible within rules: learn optimal byte-pair merges
- Higher-order n-gram mixer (orders 5-10) with PPMII escape
- Entropy-adaptive alpha tuning per-position
- Expected: 0.01-0.05 BPB

### Estimated Total
- Phase 1: ~1.0-1.15 BPB
- Phase 2: ~0.90-1.05 BPB
- Phase 3: ~0.89-1.04 BPB
- Phase 4: ~0.85-1.00 BPB

### Key Insight
The gap between 1.11 BPB and sub-1.0 is NOT about making a better neural model.
It's about making the neural model and n-gram model COMPLEMENTARY. The neural model
should never predict what n-grams already know. This is the insight behind
complementary training, but residual prediction takes it all the way.

### Why Nobody Has Done This
1. Most teams are focused on pure neural improvements (architecture, optimizer, quantization)
2. The n-gram mixer wave in late March was mostly naive mixing, not complementary
3. Residual prediction requires joint training of the n-gram component and model
4. Compression-aware training requires differentiating through a compression proxy
5. These are fundamentally different ways to think about the problem