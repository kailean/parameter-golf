#!/bin/bash
export ITERATIONS=100
export TRAIN_BATCH_TOKENS=262144
export TRAIN_LOG_EVERY=25
export VAL_LOSS_EVERY=50
export VAL_BATCH_SIZE=524288
export RUN_ID=quick_test
export DATA_PATH=/Users/kaileanhard/.openclaw/workspace/parameter-golf/data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=/Users/kaileanhard/.openclaw/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model

cd /Users/kaileanhard/.openclaw/workspace/parameter-golf
/opt/homebrew/bin/python3.11 train_gpt_mlx_kl.py 2>&1 | tee logs/quick_test_live.log
