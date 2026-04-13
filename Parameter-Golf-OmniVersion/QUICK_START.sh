#!/bin/bash
cd ~/.openclaw/workspace/parameter-golf

# Set environment
export ITERATIONS=100
export TRAIN_BATCH_TOKENS=262144
export TRAIN_LOG_EVERY=10
export VAL_LOSS_EVERY=50
export VAL_BATCH_SIZE=524288
export RUN_ID=quick_test_$(date +%H%M)
export MAX_WALLCLOCK_SECONDS=300

echo "=========================================="
echo "PARAMETER GOLF - Quick Test (100 iters)"
echo "=========================================="
echo ""
echo "Dataset: fineweb10B_sp1024"
echo "Tokens per batch: 262,144"
echo "Iterations: 100"
echo "Estimated time: 5-10 minutes"
echo ""
echo "Starting training..."
echo "=========================================="

/opt/homebrew/bin/python3.11 train_gpt_mlx_kl.py

echo ""
echo "=========================================="
echo "Training complete!"
echo "Check logs/ directory for results"
echo "=========================================="