#!/bin/bash
# Phase 1 Moonshot: Unlock all disabled innovations
# Baseline comparison: ablation_v3_baseline (all flags OFF, val_bpb 2.76)

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
export TRAIN_SEQ_LEN=1024

# ── PHASE 1: Unlock what's already there ──

# EngramLite: Gated multi-head bigram+trigram hash (replaces BigramHash)
export ENGRAM_LITE_ENABLED=1
export ENGRAM_HASH_SIZE=2048
export ENGRAM_EMBED_DIM=128
export ENGRAM_N_HEADS=2

# SkipGram hash (non-contiguous token patterns)
export SKIPGRAM_HASH_SIZE=4096

# Complementary Training: down-weight tokens predictable by n-grams
export COMPLEMENT_ALPHA=0.5

# BackoffNgramMixer at eval time (zero artifact cost)
export NGRAM_MIXER_ENABLED=1
export NGRAM_ALPHA=0.25
export NGRAM_MAX_ORDER=4

# ── Existing stack (same as v3 baseline) ──
export QAT_START_FRAC=0.15
export EMA_DECAY=0.995
export EMA_START_FRAC=0.5

# ── Eval & Output ──
export EVAL_MODE=sliding
export OUT_DIR="logs/moonshot_phase1"
export RUN_ID="moonshot_phase1"

python3 train_gpt_mlx_kl.py