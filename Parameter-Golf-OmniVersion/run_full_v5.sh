#!/bin/bash
# v5: Standard training (no residual prediction — too much RAM)
# 500 iterations, standard loss, then Dirichlet eval at the end
# This matches the smoke test config that worked before, just longer

export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"

# ── Core Architecture ──
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export MLP_MULT=3
export VOCAB_SIZE=1024

# ── Training ──
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
export WARMUP_STEPS=20
export WARMDOWN_ITERS=400

# ── Dirichlet Mixer (eval-time only, no training overhead!) ──
export DIRICHLET_MIXER=1
export DIRICHLET_MAX_ORDER=15
export DIRICHLET_ALPHA=0.25
export PHRASE_CACHE_ENABLED=1
export PHRASE_PROBE_LENGTHS="20,16"
export NGRAM_MIXER_ENABLED=1

# ── NO residual prediction (RAM issue on M4) ──
export RESIDUAL_PREDICTION=0
export COMPLEMENT_ALPHA=0

# ── Disabled ──
export COMPRESS_AWARE=0
export BYTE_WEIGHTED_LOSS=0

# ── Eval: standard + sliding with Dirichlet ──
export EVAL_MODE=both
export OUT_DIR="logs/full_run_v5"
export RUN_ID="full_v5"

python3 train_gpt_mlx_kl.py