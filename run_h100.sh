#!/bin/bash
# ============================================================
# OmniClaw H100 Competition Run — Parameter Golf
# Target: Beat 1.0734 BPB (current SOTA, PR #1530)
# Budget: Azure $200, alert at $50 remaining
# ============================================================
set -euo pipefail

# ── Budget Tracking ──
BUDGET_TOTAL=200
BUDGET_ALERT=50
START_TIME=$(date +%s)

check_budget() {
    local now=$(date +%s)
    local elapsed_h=$(( (now - START_TIME) / 3600 ))
    local spent=$(echo "$elapsed_h * 3.60" | bc -l)  # 1xH100 ~$3.60/hr
    local remaining=$(echo "$BUDGET_TOTAL - $spent" | bc -l)
    echo "💰 Budget: ~\$$(printf '%.2f' $spent) spent, ~\$$(printf '%.2f' $remaining) remaining (1×H100 estimate)"
    if (( $(echo "$remaining < $BUDGET_ALERT" | bc -l) )); then
        echo "⚠️  BUDGET ALERT: Less than \$$BUDGET_ALERT remaining! Consider stopping."
    fi
}

# ── Competition Config (matches SOTA stack) ──
export DATA_PATH="./data/datasets/fineweb10B_sp8192"
export TOKENIZER_PATH="./data/tokenizers/fineweb_8192_bpe.model"
export VOCAB_SIZE=8192
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=4              # SOTA uses 4× (hidden_dim=2048)
export TIE_EMBEDDINGS=1
export TIED_EMBED_INIT_STD=0.005
export QK_GAIN_INIT=5.25       # SOTA value
export ROPE_BASE=10000.0
export LOGIT_SOFTCAP=30.0
export SEED=${SEED:-1337}
export RUN_ID="omniclaw_h100_$(date +%Y%m%d_%H%M)"

# ── Training Schedule ──
export ITERATIONS=20000
export TRAIN_BATCH_TOKENS=786432  # SOTA: 786K tokens/step
export TRAIN_SEQ_LEN=2048
export WARMUP_STEPS=20
export WARMDOWN_ITERS=3500
export MAX_WALLCLOCK_SECONDS=600  # 10-minute competition cap

# ── Optimizer (SOTA: Muon) ──
export OPTIMIZER=muon
export MUON_MOMENTUM=0.99
export EMA_DECAY=0.9965

# ── QAT (Int6 weights, Int8 embeddings) ──
export QAT_START_FRAC=0.15

# ── Depth Recurrence (L3-5 looped 3x = ~17 virtual layers) ──
export DEPTH_RECURRENCE=1
export RECURRENCE_LAYERS="3,4,5"
export RECURRENCE_LOOPS=3

# ── Parallel Residuals (GPT-J style, L7+) ──
export PARALLEL_RESIDUALS=1
export PARALLEL_RES_START=7

# ── Partial RoPE (16/64 dims) ──
export PARTIAL_ROPE=1
export ROPE_DIM=16

# ── Our Innovation: Entropy-Weighted TTT ──
export TTT_ENABLED=1
export TTT_LR=0.001
export TTT_ENTROPY_WEIGHT=1    # Our novel contribution
export TTT_DOC_INDEPENDENT=1   # Legal: doc-independent LoRA

# ── Eval ──
export VAL_LOSS_EVERY=500
export VAL_BATCH_SIZE=524288
export TRAIN_LOG_EVERY=50
export EVAL_SEQ_LEN=2048
export EVAL_STRIDE=64
export EVAL_BATCH_SEQS=32

# ── Banking (3D weight stacking) ──
export BANKING_ENABLED=1

echo "=========================================="
echo "🐉 OmniClaw H100 Competition Run"
echo "=========================================="
echo "Config: ${NUM_LAYERS}L, ${MODEL_DIM}d, ${NUM_HEADS}H, ${NUM_KV_HEADS}KV, ${MLP_MULT}×MLP"
echo "Vocab: SP${VOCAB_SIZE} | Tokens/step: ${TRAIN_BATCH_TOKENS}"
echo "Innovations: Entropy-Weighted TTT + Depth Recurrence + Parallel Residuals"
echo "Wallclock cap: ${MAX_WALLCLOCK_SECONDS}s"
echo "Seed: ${SEED}"
echo "=========================================="

check_budget

# ── Launch ──
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
echo "🎮 GPUs detected: ${NUM_GPUS}"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching distributed training with ${NUM_GPUS} GPUs..."
    torchrun --standalone --nproc_per_node=$NUM_GPUS train_gpt_kl.py
else
    echo "Launching single-GPU training..."
    python3 train_gpt_kl.py
fi

# ── Post-Run ──
END_TIME=$(date +%s)
WALLCLOCK=$((END_TIME - START_TIME))
echo ""
echo "=========================================="
echo "🐉 Training complete! Wallclock: ${WALLCLOCK}s"
echo "=========================================="
check_budget
echo "Check logs/ for results and model checkpoints"