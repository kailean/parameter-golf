#!/bin/bash
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export NGRAM_MIXER_ENABLED=0
export EVAL_MODE=sliding
export ITERATIONS=200
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=0
export TRAIN_BATCH_TOKENS=131072
export GRAD_ACCUM_STEPS=4
export TRAIN_SEQ_LEN=1024
export SEED=1337
export OUT_DIR="logs/ablation_v3_baseline"
export RUN_ID="ablation_v3_baseline"
python3 train_gpt_mlx_kl.py
