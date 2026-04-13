#!/bin/bash
# Full Competition Run: Dirichlet 15-gram + Phrase Cache + All Innovations
# This is the configuration for 8×H100 submission

export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"

# ── Core Architecture ──
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export MLP_MULT=3
export VOCAB_SIZE=1024

# ── Training ──
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=600
export VAL_LOSS_EVERY=0
export TRAIN_BATCH_TOKENS=786432
export GRAD_ACCUM_STEPS=1
export TRAIN_SEQ_LEN=2048
export SEED=1337

# ── QAT + EMA (same as SOTA) ──
export QAT_START_FRAC=0.15
export EMA_DECAY=0.997
export EMA_START_FRAC=0.50

# ── Innovations Stack ──
export BIGRAM_HASH_SIZE=16384
export SMEAR_GATE=1
export XSA_LAST_N=4
export ORTHO_INIT=1
export LN_SCALE=1

# ── Complementary Training ──
export COMPLEMENT_ALPHA=0.5

# ── Dirichlet N-gram Mixer (eval-time) ──
export DIRICHLET_MIXER=1
export DIRICHLET_MAX_ORDER=15
export DIRICHLET_ALPHA=0.25
export PHRASE_CACHE_ENABLED=1
export PHRASE_PROBE_LENGTHS="20,16"

# ── NgramMixer off (using Dirichlet instead) ──
export NGRAM_MIXER_ENABLED=0

# ── Eval ──
export EVAL_MODE=sliding
export OUT_DIR="logs/competition_v1"
export RUN_ID="competition_v1"

# Launch on 8×H100:
# torchrun --standalone --nproc_per_node=8 train_gpt_kl.py
# On M4 (smoke test, fewer iterations):
python3 train_gpt_mlx_kl.py