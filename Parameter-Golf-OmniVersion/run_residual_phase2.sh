#!/bin/bash
# Phase 2: Residual Prediction Architecture
# The model outputs RESIDUAL logits that ADD to n-gram logits before CE loss.
# The model learns ONLY what n-grams cannot predict.

export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"

# ── Core Architecture ──
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export MLP_MULT=3
export VOCAB_SIZE=1024

# ── Training ──
export ITERATIONS=200
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=0
export TRAIN_BATCH_TOKENS=131072
export GRAD_ACCUM_STEPS=4
export TRAIN_SEQ_LEN=1024
export SEED=1337

# ── RESIDUAL PREDICTION (the new hotness) ──
export ENGRAM_LITE_ENABLED=1
export ENGRAM_HASH_SIZE=2048
export ENGRAM_EMBED_DIM=128
export ENGRAM_N_HEADS=2
export SKIPGRAM_HASH_SIZE=4096

# Residual mode: n-gram logits subtracted from target, model learns residual
export RESIDUAL_PREDICTION=1
export RESIDUAL_ALPHA=0.5

# NgramMixer at eval time
export NGRAM_MIXER_ENABLED=1
export NGRAM_ALPHA=0.25
export NGRAM_MAX_ORDER=4

# Byte-weighted loss (Q3: SP1024 has 4.49 bytes/token mean!)
export BYTE_WEIGHTED_LOSS=1

# ── Existing stack ──
export QAT_START_FRAC=0.15
export EMA_DECAY=0.995
export EMA_START_FRAC=0.5

# ── Eval & Output ──
export EVAL_MODE=sliding
export OUT_DIR="logs/residual_phase2"
export RUN_ID="residual_phase2"

python3 train_gpt_mlx_kl.py