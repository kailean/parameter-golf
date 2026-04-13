#!/bin/bash
# Phase 2a: Clean baseline with RESIDUAL prediction only
# No EngramLite, No SkipGram, No complementary training
# Just the core baseline + residual prediction at low alpha
# This isolates whether residual prediction helps on its own

export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"

# ── Core Architecture (same as v3 baseline) ──
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export MLP_MULT=3
export VOCAB_SIZE=1024

# ── Training (same as v3 baseline) ──
export ITERATIONS=200
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=0
export TRAIN_BATCH_TOKENS=131072
export GRAD_ACCUM_STEPS=4
export TRAIN_SEQ_LEN=1024
export SEED=1337

# ── RESIDUAL PREDICTION ONLY (low alpha) ──
# This adds n-gram logit bias during training but keeps everything else baseline
export RESIDUAL_PREDICTION=1
export RESIDUAL_ALPHA=0.3
export COMPLEMENT_ALPHA=0.001  # tiny, just triggers bigram stats build

# ── NO extra hash tables, NO EngramLite, NO SkipGram ──
export ENGRAM_LITE_ENABLED=0
export SKIPGRAM_HASH_SIZE=0

# ── NgramMixer at eval (same as Phase 1) ──
export NGRAM_MIXER_ENABLED=1
export NGRAM_ALPHA=0.25
export NGRAM_MAX_ORDER=4

# ── Existing stack (same as v3 baseline) ──
export QAT_START_FRAC=0.15
export EMA_DECAY=0.995
export EMA_START_FRAC=0.5

# ── Eval & Output ──
export EVAL_MODE=standard
export OUT_DIR="logs/residual_only_phase2a"
export RUN_ID="residual_only_phase2a"

python3 train_gpt_mlx_kl.py