#!/bin/bash
set -e
export DATA_PATH="./data/datasets/fineweb10B_sp1024"
export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
export NGRAM_MIXER_ENABLED=1
export NGRAM_ALPHA=0.25
export NGRAM_MAX_ORDER=4
export EVAL_MODE=sliding
export ITERATIONS=20000
export MAX_WALLCLOCK_SECONDS=600
export SEED=1337
export OUT_DIR="logs/ngram_mixer_seed1337"
export RUN_ID="ngram_mixer_seed1337"
python3 train_gpt_mlx_kl.py
