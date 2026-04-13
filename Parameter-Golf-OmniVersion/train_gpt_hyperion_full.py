#!/usr/bin/env python3
"""
HYPERION: Full training script for Parameter Golf
Sparse factorized embeddings + hierarchical attention
"""

from __future__ import annotations
import os, sys, time, math, json, glob
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Configuration
@dataclass
class Config:
    vocab_size: int = 1024
    atom_dim: int = 64
    dict_size: int = 256
    sparsity_k: int = 4
    num_layers: int = 8
    mlp_mult: int = 3
    num_heads: int = 8
    
    # Training
    iterations: int = 1000
    batch_tokens: int = 262144
    seq_len: int = 1024
    grad_accum: int = 8
    lr: float = 0.001
    warmup: int = 50
    
    # Paths
    data_path: str = "./data/datasets/fineweb10B_sp1024"
    tokenizer_path: str = "./data/tokenizers/fineweb_1024_bpe.model"
    run_id: str = "hyperion"
    
    # Eval
    val_every: int = 100
    val_tokens: int = 524288


class SparseEmbed(nn.Module):
    """Sparse factorized embedding"""
    def __init__(self, vs: int, ds: int, ad: int, k: int):
        super().__init__()
        self.dictionary = nn.Embedding(ds, ad)
        self.indices = nn.Embedding(vs, k)
        self.weights = nn.Embedding(vs, k)
        self.dictionary.weight = self.dictionary.weight * 0.02
        self.weights.weight = self.weights.weight * 0.01
    
    def __call__(self, t: mx.array) -> mx.array:
        B, T = t.shape
        idx = self.indices(t).astype(mx.int32)
        w = nn.softmax(self.weights(t), -1)
        atoms = self.dictionary(idx.reshape(-1)).reshape(B, T, -1, self.dictionary.weight.shape[1])
        return (w[..., None] * atoms).sum(axis=2)


class HierarchicalAttn(nn.Module):
    """Hierarchical attention with 4 levels"""
    def __init__(self, d: int):
        super().__init__()
        self.qkv = nn.Linear(d, d*3)
        self.phrase = nn.Linear(d, d)
        self.clause = nn.Linear(d, d)
        self.doc = nn.Linear(d, d)
        self.out = nn.Linear(d*4, d)
        self.gates = nn.Linear(d, 4)
    
    def __call__(self, x: mx.array, mask=None) -> mx.array:
        B, T, d = x.shape
        qkv = self.qkv(x).split(3, -1)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(d)
        if mask is not None: scores = scores + mask
        token = nn.softmax(scores, -1) @ v
        phrase, clause, doc = self.phrase(x), self.clause(x), self.doc(x)
        gates = nn.softmax(self.gates(x.mean(1, keepdims=True)), -1)[..., None]
        combined = mx.stack([token, phrase, clause, doc], 2)
        mixed = (combined * gates).sum(2)
        return self.out(mx.concatenate([token, phrase, clause, doc], -1))


class Block(nn.Module):
    def __init__(self, d: int, mult: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = HierarchicalAttn(d)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d*mult), nn.GELU(), nn.Linear(d*mult, d))
    
    def __call__(self, x: mx.array, mask=None) -> mx.array:
        return x + self.ffn(self.ln2(x + self.attn(self.ln1(x), mask)))


class Hyperion(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.embed = SparseEmbed(c.vocab_size, c.dict_size, c.atom_dim, c.sparsity_k)
        self.blocks = [Block(c.atom_dim, c.mlp_mult) for _ in range(c.num_layers)]
        self.ln_f = nn.LayerNorm(c.atom_dim)
        self.head = nn.Linear(c.atom_dim, c.vocab_size)
    
    def __call__(self, t: mx.array, targets=None):
        x = self.embed(t)
        mask = mx.triu(mx.full((t.shape[1], t.shape[1]), -1e9), k=1)
        for b in self.blocks: x = b(x, mask)
        logits = self.head(self.ln_f(x))
        if targets is None: return logits
        loss = nn.losses.cross_entropy(logits.reshape(-1, self.c.vocab_size), targets.reshape(-1))
        return loss, logits


def load_data(config: Config):
    """Load FineWeb tokens"""
    train_files = sorted(glob.glob(f"{config.data_path}/fineweb_train_*.bin"))
    val_file = f"{config.data_path}/fineweb_val_000000.bin"
    
    if not train_files:
        raise FileNotFoundError(f"No training files found in {config.data_path}")
    
    # Load validation
    val_tokens = np.fromfile(val_file, dtype=np.uint32).astype(np.int32)
    val_tokens = val_tokens[:config.val_tokens]
    
    return train_files, val_tokens


def get_batch(files, batch_tokens, seq_len):
    """Sample random batch from training files"""
    file = np.random.choice(files)
    tokens = np.fromfile(file, dtype=np.uint32).astype(np.int32)
    
    # Random chunk
    max_start = len(tokens) - batch_tokens - 1
    if max_start <= 0:
        return get_batch(files, batch_tokens, seq_len)
    
    start = np.random.randint(0, max_start)
    chunk = tokens[start:start + batch_tokens + 1]
    
    # Reshape to sequences
    n_seqs = batch_tokens // seq_len
    chunk = chunk[:n_seqs * seq_len + 1]
    
    x = chunk[:-1].reshape(n_seqs, seq_len)
    y = chunk[1:].reshape(n_seqs, seq_len)
    
    return mx.array(x), mx.array(y)


def evaluate(model, val_tokens, config):
    """Run validation"""
    model.eval()
    
    n_batches = len(val_tokens) // config.seq_len
    total_loss = 0.0
    total_tokens = 0
    
    for i in range(n_batches):
        start = i * config.seq_len
        end = start + config.seq_len + 1
        if end > len(val_tokens): break
        
        x = mx.array(val_tokens[start:end-1].reshape(1, -1))
        y = mx.array(val_tokens[start+1:end].reshape(1, -1))
        
        loss, _ = model(x, y)
        total_loss += float(loss) * config.seq_len
        total_tokens += config.seq_len
    
    avg_loss = total_loss / total_tokens
    bpb = avg_loss / math.log(2)
    
    model.train()
    return avg_loss, bpb


def train():
    """Main training loop"""
    config = Config()
    
    print("="*60)
    print("HYPERION - Parameter Golf Training")
    print("="*60)
    print(f"Run ID: {config.run_id}")
    print(f"Data: {config.data_path}")
    
    # Load data
    print("\nLoading data...")
    try:
        train_files, val_tokens = load_data(config)
        print(f"✓ Train files: {len(train_files)}")
        print(f"✓ Val tokens: {len(val_tokens):,}")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return
    
    # Build model
    print("\nBuilding HYPERION model...")
    model = Hyperion(config)
    
    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    print(f"Parameters: {n_params:,} (~{n_params*4/1024/1024:.1f} MB)")
    
    # Optimizer
    opt = optim.Adam(learning_rate=config.lr)
    
    # Training loop
    print(f"\nTraining {config.iterations} iterations...")
    print("-"*60)
    
    losses = []
    best_bpb = float('inf')
    start_time = time.time()
    
    for step in range(config.iterations):
        # Get batch
        x, y = get_batch(train_files, config.batch_tokens, config.seq_len)
        
        # Forward
        loss, _ = model(x, y)
        
        # Backward
        loss_value = float(loss.mean())
        losses.append(loss_value)
        
        # Update
        def loss_fn(m):
            l, _ = m(x, y)
            return l.mean()
        grads = mx.grad(loss_fn)(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        
        # Log
        if step % 10 == 0:
            avg_loss = sum(losses[-10:]) / len(losses[-10:])
            print(f"Step {step:4d}/{config.iterations} | Loss: {avg_loss:.4f}")
        
        # Validate
        if step > 0 and step % config.val_every == 0:
            print(f"\n[Validating at step {step}...]")
            val_loss, val_bpb = evaluate(model, val_tokens, config)
            print(f"Validation | Loss: {val_loss:.4f} | BPB: {val_bpb:.4f}")
            
            if val_bpb < best_bpb:
                best_bpb = val_bpb
                print(f"★ New best BPB: {best_bpb:.4f}")
            print()
    
    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    final_loss, final_bpb = evaluate(model, val_tokens, config)
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Final BPB:  {final_bpb:.4f}")
    print(f"Best BPB:   {best_bpb:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining time: {elapsed/60:.1f} minutes")
    
    # Save results
    results = {
        "run_id": config.run_id,
        "final_bpb": float(final_bpb),
        "best_bpb": float(best_bpb),
        "params": n_params,
        "time_seconds": elapsed
    }
    
    out_file = f"logs/{config.run_id}_results.json"
    os.makedirs("logs", exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {out_file}")


if __name__ == "__main__":
    train()
