#!/usr/bin/env python3
"""
Quick Training + Evaluation for Parameter Golf
Trains briefly, saves checkpoint, then runs proper eval.
"""
import os, sys, time, math

# Set quick mode
os.environ["ITERATIONS"] = "100"
os.environ["VAL_LOSS_EVERY"] = "0"
os.environ["TRAIN_BATCH_TOKENS"] = "262144"
os.environ["TRAIN_LOG_EVERY"] = "25"
os.environ["MAX_WALLCLOCK_SECONDS"] = "300"  # 5 min max
os.environ["RUN_ID"] = "quick_eval"

print("="*60)
print("PARAMETER GOLF - Quick Train + Eval")
print("="*60)
print()

# Import and run training
print("Step 1: Training for 100 iterations...")
print("-" * 60)
exec(open('train_gpt_mlx_kl.py').read())

# After training, run eval
print()
print("="*60)
print("Step 2: Running validation...")
print("="*60)

# The train script should have saved a checkpoint
import glob
from pathlib import Path

log_dir = Path("logs")
checkpoints = list(log_dir.glob("*_model.npz"))
if checkpoints:
    checkpoint_path = str(max(checkpoints, key=lambda p: p.stat().st_mtime))
    print(f"Found checkpoint: {checkpoint_path}")
    
    # Run eval
    os.environ["CHECKPOINT"] = checkpoint_path
    exec(open('eval_mlx.py').read())
else:
    print("ERROR: No checkpoint found after training!")
    print("Check logs/ directory for output files.")
