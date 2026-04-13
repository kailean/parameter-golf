#!/bin/bash
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export NGRAM_MIXER_ENABLED=0
export EVAL_MODE=sliding
export ITERATIONS=500
export MAX_WALLCLOCK_SECONDS=0
export VAL_LOSS_EVERY=500
export TRAIN_BATCH_TOKENS=524288
export GRAD_ACCUM_STEPS=8
export SEED=1337
export OUT_DIR="logs/ablation_baseline"
export RUN_ID="ablation_baseline_seed1337"
python3 train_gpt_mlx_kl.py
