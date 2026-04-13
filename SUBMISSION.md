# Record: SP8192 + GQA + Depth Recurrence + Mixed Int8/Int6 Quant + Score-First TTT

**val_bpb = ~1.16** (WIP, 1-seed) | **~14.8 MB** | 8×H100 SXM

> ⚠️ This is a work-in-progress submission. Training is ongoing and we expect significant BPB improvements as we complete full 20K-step training runs. Current result is from early stopping (step 1000 on 1×H100, step 4000 on 8×H100).

## Results (Preliminary)

| Config | Step | val_bpb | Artifact Size | Notes |
|--------|------|---------|---------------|-------|
| 1×H100, 30M params | 1000 | 1.1788 | ~7.8 MB | TTT disabled, MLP_MULT=3 |
| 8×H100, 30M params | 4000 | 1.1559 | ~7.8 MB | QAT NaN at step 4058 (DDP bug) |
| 1×H100, 55M params | 200 | 4.8912 (train) | TBD | Smoke test, QAT disabled |

Merged SOTA (PR #1530): **1.0734 BPB**. Delta: **+0.0825** (WIP — expect to close gap with full training).

## Novel Contributions

1. **Mixed Int8/Int6 + Brotli Quantization** — Embedding matrices (tok_emb/lm_head) are quantized to int8, all other weights to int6 packed per-row. This preserves embedding quality better than uniform int6 while maintaining compression. Combined with GPTQ-lite per-row clip percentile search for optimal quantization. First submission using this mixed approach.

2. **Score-First TTT with Rollback** — Doc-independent LoRA TTT on Q/V projections at eval time. Novel: we compute baseline loss *before* TTT adaptation and *rollback* LoRA weights if TTT makes the loss worse. Previous TTT implementations (PR #1477, #1413) always keep TTT adaptations; our approach is more conservative and avoids the degradation we observed with entropy-weighted TTT (1.73 vs 1.18 BPB).

3. **GQA with Depth Recurrence** — Grouped Query Attention (10 query heads, 5 KV heads) reduces KV cache and attention FLOPs. Combined with depth recurrence on layers 3-5 (2× loops), yielding 14 effective forward passes from 11 physical layers with zero extra parameters.

## Full Architecture Stack

| Component | Value | Source/Inspiration |
|-----------|-------|--------------------|
| Tokenizer | SP8192 (FineWeb 8192 BPE) | PR #1394 @clarkkev |
| Model dim | 640 | — |
| Layers | 11 | — |
| Heads (Q/KV) | 10 / 5 (GQA) | — |
| MLP mult | 4.0 | PR #1218 @clarkkev |
| Tied embeddings | Yes, init_std=0.005 | — |
| Depth recurrence | L3-5, 2× loops (17 virtual layers from 11) | PR #1334, PR #1394 |
| Parallel residuals | L7+ (GPT-J style) | PR #1204 @msisovic, PR #1412 @Robby955 |
| QK-Gain | init=5.25 | PR #1493 @bigbag |
| Partial RoPE | dim=16 | — |
| SmearGate | Blend token embedding with predecessor's | — |
| Logit softcap | 30.0 | — |
| XSA | Cross-sequence attention, last 4 layers (L7-L10) | — |
| Ortho init | Enabled | — |

## Quantization Stack

| Component | Format | Details |
|-----------|--------|---------|
| Embeddings | int8 per-row | tok_emb.weight, lm_head.weight |
| Weights | int6 packed per-row | 4 values per 3 bytes |
| Clip calibration | GPTQ-lite | Search 5 percentiles per row, minimize MSE |
| Compression | Brotli quality=11 | Best ratio for int6 packed data |
| Code wrapper | LZMA | Minimize code footprint |
| **Total artifact** | **~14.8 MB** | **Under 16 MB** ✅ |

## Training Stack

| Parameter | Value |
|-----------|-------|
| Optimizer | Muon (momentum=0.99) |
| EMA | decay=0.9965, start at 50% |
| Weight decay | 0.095 |
| Batch tokens | 786,432/step |
| Seq len | 2048 |
| Warmup | 20 steps |
| Warmdown | Last 3500 iterations |
| QAT | Disabled (LATE_QAT_THRESHOLD=0.0) |
| Post-hoc quant | GPTQ-lite |

## Compliance (Track B)

Per Issue #1017:
- **Condition 1 (Causality):** Sliding-window eval, prefix only ✅
- **Condition 2 (Normalized):** Standard softmax, no n-gram/logit bias ✅
- **Condition 3 (Score before update):** Each chunk scored under `torch.no_grad()` BEFORE TTT adaptation ✅
- **Condition 4 (Single pass):** Each token scored once, no rescoring ✅
- **Rollback:** TTT adaptations are reverted if they increase loss (more conservative than always-keep)

No SLOT, no pre-quant TTT, no ETLB, no n-gram cache. All artifacts < 16MB, train < 600s, eval < 600s.

## Reproduction

```bash
# 1. Download data
pip install brotli sentencepiece huggingface_hub
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --skip-manifest

# 2. Train (8×H100)
SEED=42 \
MODEL_DIM=640 NUM_LAYERS=11 NUM_HEADS=10 NUM_KV_HEADS=5 MLP_MULT=4 \
VOCAB_SIZE=8192 TIE_EMBEDDINGS=1 TIED_EMBED_INIT_STD=0.005 \
QK_GAIN_INIT=5.25 ROPE_BASE=10000.0 ROPE_DIMS=16 LOGIT_SOFTCAP=30.0 \
SMEAR_ENABLED=1 LN_SCALE_ENABLED=1 XSA_LAST_N=4 USE_ORTHO_INIT=1 \
DEPTH_RECURRENCE=1 RECURRENCE_LAYERS=3,4,5 RECURRENCE_LOOPS=2 \
PARALLEL_RESIDUALS=1 PARALLEL_RES_START=7 \
BIGRAM_HASH_SIZE=0 EMBED_BITS=8 \
TTT_ENABLED=0 \
LATE_QAT_THRESHOLD=0.0 USE_GPTQ_LITE=1 \
OPTIMIZER=muon MUON_MOMENTUM=0.99 EMA_DECAY=0.9965 \
ITERATIONS=20000 TRAIN_BATCH_TOKENS=786432 WARMUP_STEPS=20 WARMDOWN_ITERS=3500 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt_kl.py
```

## Credits

PR #1394 @clarkkev (SP8192, GPTQ, depth recurrence), PR #1493 @bigbag (QK-Gain 5.25, 3-layer recurrence), PR #1477 @aryanbhosale (parallel residuals + TTT), PR #1413 @dexhunter (score-first TTT), PR #1412 @Robby955 (parallel residuals), PR #1204 @msisovic (parallel residuals), PR #1530 @samacqua (VarLen + fused MLP + doc-independent TTT)

## Acknowledgements

Work in progress — seeking compute credits to complete full training runs. Applied for OpenAI's Advanced Competitor grant ($1,000 / ~320 H100 hours via RunPod).