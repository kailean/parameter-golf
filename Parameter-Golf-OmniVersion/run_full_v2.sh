#!/bin/bash
# Fast Full Run v2 — Residual Prediction + Dirichlet Mixer
# No compress_aware (too slow, minimal gain), no byte_weighted (needs adaptive_vocab.py)
# Focus: proper training with residual loss, then Dirichlet eval

export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"

# ── Core Architecture ──
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export MLP_MULT=3
export VOCAB_SIZE=1024

# ── Training ──
export ITERATIONS=10000
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=2000
export TRAIN_BATCH_TOKENS=524288
export GRAD_ACCUM_STEPS=8
export TRAIN_SEQ_LEN=1024
export SEED=1337
export TRAIN_LOG_EVERY=50

# ── Training stack ──
export QAT_START_FRAC=0.15
export EMA_DECAY=0.995
export EMA_START_FRAC=0.5

# ── Dirichlet Mixer (eval-time) ──
export DIRICHLET_MIXER=1
export DIRICHLET_MAX_ORDER=15
export DIRICHLET_ALPHA=0.25
export PHRASE_CACHE_ENABLED=1
export PHRASE_PROBE_LENGTHS="20,16"

# ── NgramMixer must be ON for Dirichlet to activate at eval ──
export NGRAM_MIXER_ENABLED=1

# ── Residual Prediction (key innovation!) ──
export RESIDUAL_PREDICTION=1
export COMPLEMENT_ALPHA=0.001

# ── DISABLED: compress_aware too slow, minimal gain ──
export COMPRESS_AWARE=0
export BYTE_WEIGHTED_LOSS=0

# ── Eval & Output ──
export EVAL_MODE=both
export OUT_DIR="logs/full_run_v2"
export RUN_ID="full_v2"

python3 train_gpt_mlx_kl.py