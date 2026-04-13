#!/bin/bash
# Fast Smoke Test v4 — Maximum speed, all key innovations
# - 500 iterations (enough to converge + see BPB trend)
# - No warmup (skip 20 wasted steps)
# - Smaller batch (131K tokens vs 524K — 4x faster per step)
# - Residual Prediction + Dirichlet Mixer
# - Standard eval only (skip slow sliding eval for now, do it after)

export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"

# ── Core Architecture ──
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export MLP_MULT=3
export VOCAB_SIZE=1024

# ── Training: fast config ──
export ITERATIONS=500
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=100
export TRAIN_BATCH_TOKENS=131072
export GRAD_ACCUM_STEPS=4
export TRAIN_SEQ_LEN=1024
export SEED=1337
export TRAIN_LOG_EVERY=10

# ── Training stack ──
export QAT_START_FRAC=0.15
export EMA_DECAY=0.995
export EMA_START_FRAC=0.5
export WARMUP_STEPS=0
export WARMDOWN_ITERS=400

# ── Dirichlet Mixer (eval-time) ──
export DIRICHLET_MIXER=1
export DIRICHLET_MAX_ORDER=15
export DIRICHLET_ALPHA=0.25
export PHRASE_CACHE_ENABLED=1
export PHRASE_PROBE_LENGTHS="20,16"
export NGRAM_MIXER_ENABLED=1

# ── Residual Prediction ──
export RESIDUAL_PREDICTION=1
export COMPLEMENT_ALPHA=0.001

# ── Disabled ──
export COMPRESS_AWARE=0
export BYTE_WEIGHTED_LOSS=0

# ── Eval: standard only (fast), Dirichlet eval separately after ──
export EVAL_MODE=standard
export OUT_DIR="logs/full_run_v4"
export RUN_ID="full_v4"

python3 train_gpt_mlx_kl.py