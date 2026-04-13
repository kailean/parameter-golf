#!/bin/bash
# OmniClaw Parameter Golf Training Script
# Target: Beat 1.0734 BPB (current SOTA)
# Stack: SP8192 + GPTQ SDClip + Depth Recurrence + Parallel Residuals + Legal TTT
# Hardware: 8×H100 SXM (RunPod) or M4 (smoke test)

set -euo pipefail

export VOCAB_SIZE=8192
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4
export TIE_EMBEDDINGS=1
export TIED_EMBED_INIT_STD=0.005
export ROPE_BASE=10000.0
export LOGIT_SOFTCAP=30.0

# Training
export ITERATIONS=20000
export WARMUP_STEPS=20
export WARMDOWN_ITERS=3500
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024
export MAX_WALLCLOCK_SECONDS=600

# Muon optimizer
export MATRIX_LR=0.022
export SCALAR_LR=0.02
export EMBED_LR=0.6
export TIED_EMBED_LR=0.05
export MUON_MOMENTUM=0.99
export MUON_BACKEND_STEPS=5
export MUON_MOMENTUM_WARMUP_START=0.85
export MUON_MOMENTUM_WARMUP_STEPS=500
export WEIGHT_DECAY=0.095

# QK-Gain
export QK_GAIN_INIT=5.25

# EMA
export USE_EMA=1
export EMA_DECAY=0.9965

# Depth recurrence (L3-5, looped 3x = 17 virtual layers from 11 physical)
export RECURRENCE_LAYERS="3,4,5"
export RECURRENCE_COUNT=3
export RECURRENCE_START_STEP=3000

# Parallel residuals (L7+)
export PARALLEL_RESIDUAL_FROM=7

# Legal score-first TTT
export TTT_LORA_RANK=96
export TTT_LORA_LR=0.01
export TTT_CHUNK_SIZE=256
export TTT_EVAL_SEQ_LEN=1024
export TTT_BATCH_SIZE=64
export TTT_SCORE_FIRST=1

# Data paths
export DATA_PATH="./data/datasets/fineweb10B_sp8192"
export TOKENIZER_PATH="./data/tokenizers/fineweb_8192_bpe.model"

# GPTQ SDClip quantization
export QUANTIZE_INT6=1
export GPTQ_SDCLIP_K=12.85
export GPTQ_EMBED_K=20.0

echo "=== OmniClaw Training Config ==="
echo "Target: Beat 1.0734 BPB"
echo "Vocab: $VOCAB_SIZE | Layers: $NUM_LAYERS | Dim: $MODEL_DIM"
echo "Optimizer: Muon (momentum=$MUON_MOMENTUM)"
echo "QK-Gain: $QK_GAIN_INIT | EMA: $EMA_DECAY"
echo "Recurrence: L${RECURRENCE_LAYERS} x${RECURRENCE_COUNT} (start step $RECURRENCE_START_STEP)"
echo "TTT: Legal score-first, rank=$TTT_LORA_RANK"
echo "================================"

# Run training
python3 train_gpt.py "$@"