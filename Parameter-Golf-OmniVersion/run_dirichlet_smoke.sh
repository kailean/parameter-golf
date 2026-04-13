#!/bin/bash
# Dirichlet Mixer Smoke Test
# Tests the DirichletNgramMixer with orders 2-15, phrase cache, OBCL concentrations
# Uses only 1 training shard for fast iteration on M4

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

# ── Existing stack (same as v3 baseline) ──
export QAT_START_FRAC=0.15
export EMA_DECAY=0.995
export EMA_START_FRAC=0.5

# ── DIRICHLET MIXER (eval-time only, no training overhead) ──
export DIRICHLET_MIXER=1
export DIRICHLET_MAX_ORDER=15
export DIRICHLET_ALPHA=0.25
export PHRASE_CACHE_ENABLED=1
export PHRASE_PROBE_LENGTHS="20,16"

# ── Keep NgramMixer off (we're using Dirichlet instead) ──
export NGRAM_MIXER_ENABLED=0

# ── Eval & Output ──
export EVAL_MODE=sliding
export OUT_DIR="logs/dirichlet_smoke"
export RUN_ID="dirichlet_smoke"

python3 train_gpt_mlx_kl.py