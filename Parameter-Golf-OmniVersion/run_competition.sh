#!/bin/bash
# M4 Competition Training Run
# Full training with complementary training enabled
# Wallclock-limited to 6 hours (21600 seconds)
# Dirichlet mixer OFF during training (it's eval-only)
# After training: run eval_dirichlet_only.py for final BPB

export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"

# ── Architecture ──
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3
export VOCAB_SIZE=1024

# ── Training ──
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=21600  # 6 hours
export VAL_LOSS_EVERY=0  # No intermediate val (saves time)
export TRAIN_BATCH_TOKENS=524288
export GRAD_ACCUM_STEPS=8
export TRAIN_SEQ_LEN=1024
export TRAIN_LOG_EVERY=50
export SEED=1337

# ── Training Innovations ──
export QAT_START_FRAC=0.15
export EMA_DECAY=0.995
export EMA_START_FRAC=0.5
export USE_ORTHO_INIT=1
export USE_SMEARGATE=1
export ROPE_DIMS=16
export LN_SCALE_ENABLED=1
export XSA_LAST_N=4
export USE_GPTQ_LITE=1

# ── Complementary Training (KEY: makes model focus on what n-grams miss) ──
export COMPLEMENT_ALPHA=0.50

# ── Dirichlet OFF during training (eval-only) ──
export DIRICHLET_MIXER=0
export NGRAM_MIXER_ENABLED=0

# ── Eval & Output ──
export EVAL_MODE=both  # Standard + sliding at end
export OUT_DIR="logs/competition_v1"
export RUN_ID="competition_v1"

python3 train_gpt_mlx_kl.py