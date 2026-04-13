#!/usr/bin/env python3
"""Simple HYPERION training - working version"""
import sys, time, math, json
sys.path.insert(0, '.')

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from train_gpt_hyperion_full import Hyperion, Config

def count_params(model):
    return sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))

def quick_train(iterations=30):
    print("="*60)
    print("HYPERION Quick Training")
    print("="*60)
    
    config = Config()
    config.iterations = iterations
    
    # Build model
    print("\nBuilding HYPERION...")
    model = Hyperion(config)
    n_params = count_params(model)
    print(f"Parameters: {n_params:,} (~{n_params*4/1024/1024:.1f}MB)")
    
    # Load data
    print("\nLoading data...")
    import glob
    train_files = sorted(glob.glob(f"{config.data_path}/fineweb_train_*.bin"))[:5]
    val_file = f"{config.data_path}/fineweb_val_000000.bin"
    val_tokens = np.fromfile(val_file, dtype=np.uint32).astype(np.int32)[:50000]
    print(f"Train files: {len(train_files)}, Val tokens: {len(val_tokens):,}")
    
    # Optimizer
    opt = optim.Adam(learning_rate=0.001)
    
    # Train
    print(f"\nTraining {iterations} steps...")
    print("-"*60)
    
    for step in range(iterations):
        # Sample batch
        file = np.random.choice(train_files)
        tokens = np.fromfile(file, dtype=np.uint32).astype(np.int32)
        
        start = np.random.randint(0, max(1, len(tokens) - 1025))
        chunk = tokens[start:start+1025]
        
        x = mx.array(chunk[:-1].reshape(1, -1))
        y = mx.array(chunk[1:].reshape(1, -1))
        
        # Forward + backward
        def loss_fn(m):
            l, _ = m(x, y)
            return l.mean()
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        
        if step % 10 == 0:
            print(f"Step {step:2d}/{iterations} | Loss: {float(loss.mean()):.4f}")
    
    # Evaluate
    print("\nEvaluating...")
    model.eval()
    
    n_batches = len(val_tokens) // 1024
    total_loss = 0.0
    
    for i in range(min(n_batches, 20)):
        start = i * 1024
        x = mx.array(val_tokens[start:start+1024].reshape(1, -1))
        y = mx.array(val_tokens[start+1:start+1025].reshape(1, -1))
        
        loss, _ = model(x, y)
        total_loss += float(loss.mean()) * 1024
    
    avg_loss = total_loss / (min(n_batches, 20) * 1024)
    bpb = avg_loss / math.log(2)
    
    print(f"\n{'='*60}")
    print("HYPERION Results")
    print(f"{'='*60}")
    print(f"Parameters: {n_params:,}")
    print(f"Val Loss:   {avg_loss:.4f}")
    print(f"Val BPB:    {bpb:.4f}")
    print(f"Size:       ~{n_params*4/1024/1024:.1f} MB")
    print(f"{'='*60}")
    
    return {
        'name': 'HYPERION',
        'params': n_params,
        'bpb': bpb,
        'loss': avg_loss
    }

if __name__ == "__main__":
    results = quick_train(30)
    print(f"\nFinal BPB: {results['bpb']:.4f}")
