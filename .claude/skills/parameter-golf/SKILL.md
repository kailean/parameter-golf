# Parameter Golf Competition Skill

## Overview
This skill provides grounded knowledge for the OpenAI Parameter Golf challenge. All facts below are sourced from the official repo, verified PRs, and measured experiments.

## Competition Rules (Immutable)

The artifact limit is exactly 16,000,000 bytes (decimal, not MiB). This includes both the training code (`train_gpt.py` and any dependencies) and the compressed model weights. Training must complete within 10 minutes wall-clock on 8×H100 SXM GPUs. Evaluation gets a separate 10-minute budget. No network calls are allowed during evaluation, and the artifact must be fully self-contained.

The metric is bits-per-byte (val_bpb) on the FineWeb validation set (fixed first-50k-document slice). This is tokenizer-agnostic, so changing your tokenizer doesn't give a free advantage. New record submissions must beat the current SOTA by at least 0.005 nats and demonstrate statistical significance at p < 0.01 (typically 3 seeds).

Test-time training is allowed but ONLY on tokens you've already evaluated (backward-looking, score-first). You cannot train on validation data before scoring it.

## Verified Techniques (from top submissions)

**Quantization-Aware Training (QAT):** Nearly every competitive submission uses QAT with straight-through estimator (STE) gradients. Training the model to be robust to low-precision weights is one of the highest-leverage moves.

**int6 per-row quantization:** The standard for competitive submissions. 64 levels per row with learned or searched clip factors. GPTQ-lite clip search combined with QAT at ~0.15 weight is common.

**Depth recurrence:** Reusing transformer layers to get more effective depth without extra parameters. Key finding: unique capacity matters more than loop depth (5×2 beats 4×3 at equal params).

**Parallel residuals:** Starting at a mid-layer, attention and MLP compute in parallel rather than sequential. Saves a LayerNorm and reduces sequential dependency.

**XSA (Cross-Self-Attention):** From arXiv:2603.09078. Removes self-value bias from attention output via orthogonal projection. Applied to last 3-4 layers only. Zero parameters, minimal overhead. Near-universal in frontier submissions.

**LeakyReLU²:** Using LeakyReLU(slope=0.5) squared as activation instead of GELU. Small but consistent improvement (~0.003 bpb from ablation).

**Sliding window evaluation:** Stride-64 sliding window at eval time with seq_len=2048. Gives each token more context. Produces better bpb than fixed-length eval.

**EMA + SWA:** Exponential moving average of weights starting mid-training, stochastic weight averaging starting later. Both verified beneficial for final eval bpb.

**Parallel Muon optimizer:** Batched bank Muon optimizer with torch.compile. Co-optimized with the rest of the training stack for maximum throughput.

**SP8192 tokenizer:** 8192-token SentencePiece vocabulary. Larger than the baseline 1024 but provides better compression per byte.

## Known Failure Modes

**EMA too early:** Starting EMA from step 0 averages poorly-converged early weights. Must start mid-training (e.g., step 2250 of 4500).

**Byte shuffle with int6:** Byte-shuffling packed int6 data before brotli compression actively HURTS compression ratio by 25-75%. The int6 packing already creates well-structured data that brotli can exploit; shuffling destroys those patterns.

**Novel architectures without throughput co-optimization:** PR #831's systematic evaluation showed that 6 architectural innovations from March 2026 papers ALL failed on the SOTA stack because they broke the throughput-quantization co-optimization. Each ms/step of overhead costs ~7 steps, and each step is worth ~0.001 bpb.

**GPTQ weight corruption with TTT:** Full GPTQ can corrupt weights in ways that interact badly with test-time training. GPTQ-lite is safer for TTT-compatible models.

## Our Measured Results

These are from actual Modal runs, not estimates:

```
Run: 768d/11L/4×H100/4500 steps
- val_bpb: 1.0882 (SWA, step 4500)
- Artifact: 42.2 MB (int6+brotli, with byte shuffle)
- Status: OVER BUDGET by 2.6×
- Throughput: ~488ms/step

Compression variants (same trained model):
- Raw torch: 284.4 MB
- int8+zlib: 53.4 MB
- int6+brotli: 42.2 MB (4.96× ratio)
- int6+zstd: 43.3 MB

Key insight: 77.7M params CANNOT fit in 16MB at any quantization.
```

## Decision Framework

When evaluating whether to try a technique:

1. Does it improve bpb by more than 0.007 per ms/step of overhead? (throughput bar)
2. Does it interact well with int6 quantization? (some techniques break under quantization)
3. Does it require calibration data for GPTQ? (adds complexity and potential bugs)
4. Is it compatible with depth recurrence? (some techniques assume fixed depth)
5. Has it been demonstrated in a record-track submission? (10min/8×H100 constraint)
