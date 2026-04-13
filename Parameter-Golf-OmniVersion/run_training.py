#!/usr/bin/env python3
"""
Reliable training runner with live output
"""
import os
import sys
import subprocess
import time
from pathlib import Path

# Setup paths
base_dir = Path("/Users/kaileanhard/.openclaw/workspace/parameter-golf")
os.chdir(base_dir)

# Ensure logs directory exists
log_dir = base_dir / "logs"
log_dir.mkdir(exist_ok=True)

# Set environment variables
env = os.environ.copy()
env["ITERATIONS"] = "100"
env["TRAIN_BATCH_TOKENS"] = "262144"
env["TRAIN_LOG_EVERY"] = "10"
env["VAL_LOSS_EVERY"] = "50"
env["VAL_BATCH_SIZE"] = "524288"
env["RUN_ID"] = f"quick_test_{int(time.time())}"
env["MAX_WALLCLOCK_SECONDS"] = "600"
env["DATA_PATH"] = "/Users/kaileanhard/.openclaw/workspace/parameter-golf/data/datasets/fineweb10B_sp1024"
env["TOKENIZER_PATH"] = "/Users/kaileanhard/.openclaw/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"
env["PYTHONUNBUFFERED"] = "1"

log_file = log_dir / f"{env['RUN_ID']}.log"

print("=" * 60)
print("PARAMETER GOLF - Quick Test Training")
print("=" * 60)
print(f"Run ID: {env['RUN_ID']}")
print(f"Log file: {log_file}")
print(f"Iterations: {env['ITERATIONS']}")
print(f"Dataset: {env['DATA_PATH']}")
print("=" * 60)
print()

# Run training with unbuffered output
process = subprocess.Popen(
    ["/opt/homebrew/bin/python3.11", "-u", "train_gpt_mlx_kl.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
    universal_newlines=True,
    env=env
)

# Stream output in real-time
with open(log_file, "w") as f:
    for line in process.stdout:
        print(line, end="")
        f.write(line)
        f.flush()

process.wait()

print()
print("=" * 60)
if process.returncode == 0:
    print("✅ Training completed successfully!")
else:
    print(f"❌ Training failed with code {process.returncode}")
print(f"Log saved to: {log_file}")
print("=" * 60)

sys.exit(process.returncode)
