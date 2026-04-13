#!/usr/bin/env python3
"""
Simple evaluation script for Parameter Golf
Tests if the setup works and reports basic metrics
"""
import os
import sys
import time
import numpy as np

# Setup paths
os.chdir("/Users/kaileanhard/.openclaw/workspace/parameter-golf")
os.environ["DATA_PATH"] = "/Users/kaileanhard/.openclaw/workspace/parameter-golf/data/datasets/fineweb10B_sp1024"
os.environ["TOKENIZER_PATH"] = "/Users/kaileanhard/.openclaw/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"

print("=" * 60)
print("PARAMETER GOLF - Setup Verification")
print("=" * 60)
print()

# Test 1: Check dataset
print("Test 1: Checking dataset...")
dataset_path = os.environ["DATA_PATH"]
tokenizer_path = os.environ["TOKENIZER_PATH"]

import glob
train_files = glob.glob(f"{dataset_path}/fineweb_train_*.bin")
val_files = glob.glob(f"{dataset_path}/fineweb_val_*.bin")

print(f"  Train files found: {len(train_files)}")
print(f"  Val files found: {len(val_files)}")

if len(train_files) == 0 or len(val_files) == 0:
    print("  ❌ Dataset not found!")
    sys.exit(1)

import os as os2
train_size = sum(os2.path.getsize(f) for f in train_files)
val_size = sum(os2.path.getsize(f) for f in val_files)

print(f"  Train data: {train_size / 1024 / 1024:.1f} MB")
print(f"  Val data: {val_size / 1024 / 1024:.1f} MB")
print("  ✅ Dataset OK")
print()

# Test 2: Check tokenizer
print("Test 2: Checking tokenizer...")
import sentencepiece as spm

try:
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    vocab_size = sp.vocab_size()
    print(f"  Tokenizer loaded: {vocab_size} vocab size")
    print("  ✅ Tokenizer OK")
except Exception as e:
    print(f"  ❌ Tokenizer error: {e}")
    sys.exit(1)

print()

# Test 3: Test MLX import
print("Test 3: Checking MLX...")
try:
    import mlx.core as mx
    import mlx.nn as nn
    print(f"  MLX version: working")
    
    # Quick test
    a = mx.array([1.0, 2.0, 3.0])
    b = mx.array([4.0, 5.0, 6.0])
    c = a + b
    print(f"  MLX test: {c}")
    print("  ✅ MLX OK")
except Exception as e:
    print(f"  ❌ MLX error: {e}")
    sys.exit(1)

print()

# Test 4: Try loading training module
print("Test 4: Loading training module...")
try:
    # Import without running main
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_gpt", "train_gpt_mlx_kl.py")
    train_module = importlib.util.module_from_spec(spec)
    
    # Just compile to check for syntax errors
    with open("train_gpt_mlx_kl.py", "r") as f:
        compile(f.read(), "train_gpt_mlx_kl.py", "exec")
    
    print("  Module syntax: OK")
    print("  ✅ Training module OK")
except Exception as e:
    print(f"  ❌ Module error: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("✅ ALL TESTS PASSED - Setup is ready!")
print("=" * 60)
print()
print("Next steps:")
print("  1. Run: python3 run_training.py")
print("  2. Or manually: python3 -u train_gpt_mlx_kl.py")
print("  3. Check logs/ directory for output")
print()
