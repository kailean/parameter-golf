#!/bin/bash
# Fast Smoke Test v3 — 2000 iterations, all key innovations
# Residual Prediction + Dirichlet Mixer (eval-time only)
# No compress_aware (too slow, ~0.005 BPB gain), no byte_weighted
# Goal: get a new lowest BPB score ASAP

export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"

# ── Core Architecture ──
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export MLP_MULT=3
export VOCAB_SIZE=1024

# ── Training: 2000 iters for fast turnaround ──
export ITERATIONS=2000
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=500
export TRAIN_BATCH_TOKENS=524288
export GRAD_ACCUM_STEPS=8
export TRAIN_SEQ_LEN=1024
export SEED=1337
export TRAIN_LOG_EVERY=20

# ── Training stack ──
export QAT_START_FRAC=0.15
export EMA_DECAY=0.995
export EMA_START_FRAC=0.5
export WARMUP_STEPS=20
export WARMDOWN_ITERS=1500

# ── Dirichlet Mixer (eval-time only) ──
export DIRICHLET_MIXER=1
export DIRICHLET_MAX_ORDER=15
export DIRICHLET_ALPHA=0.25
export PHRASE_CACHE_ENABLED=1
export PHRASE_PROBE_LENGTHS="20,16"

# ── NgramMixer ON for Dirichlet eval ──
export NGRAM_MIXER_ENABLED=1

# ── Residual Prediction ──
export RESIDUAL_PREDICTION=1
export COMPLEMENT_ALPHA=0.001

# ── Disabled (too slow or needs extra deps) ──
export COMPRESS_AWARE=0
export BYTE_WEIGHTED_LOSS=0

# ── Eval & Output ──
export EVAL_MODE=both
export OUT_DIR="logs/full_run_v3"
export RUN_ID="full_v3"

python3 train_gpt_mlx_kl.py