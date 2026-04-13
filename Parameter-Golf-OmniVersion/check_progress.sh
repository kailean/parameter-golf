#!/bin/bash
# Monitor training progress

echo "Monitoring training progress..."
echo ""

LOG_DIR="/Users/kaileanhard/.openclaw/workspace/parameter-golf/logs"

# Find the latest log
LATEST_LOG=$(ls -t $LOG_DIR/*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No log file yet. Waiting for training to start..."
    exit 1
fi

echo "Latest log: $LATEST_LOG"
echo ""
echo "Last 20 lines:"
echo "---"
tail -20 "$LATEST_LOG"
echo "---"

# Check if training is still running
if pgrep -f "train_gpt_mlx_kl.py" > /dev/null; then
    echo ""
    echo "✅ Training is RUNNING"
else
    echo ""
    echo "⚠️ Training process not found"
fi
