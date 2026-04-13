#!/usr/bin/env python3
"""
Baseline Comparison: HYPERION vs Standard
Quick training run to compare BPB and parameter efficiency
"""

import sys, time, math
sys.path.insert(0, '.')

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from train_gpt_hyperion_full import Hyperion, Config as HyperionConfig, evaluate, load_data


class StandardEmbed(nn.Module):
    """Standard dense embedding for baseline"""
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.weight = nn.Embedding(vocab, dim)
    
    def __call__(self, t):
        return self.weight(t)


class StandardAttn(nn.Module):
    """Standard multi-head attention"""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
    
    def __call__(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, D // self.heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Transpose for attention: (B, H, T, D/H)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(D // self.heads)
        if mask is not None: scores = scores + mask
        attn = nn.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.out(out)


class StandardBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mult: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = StandardAttn(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim)
        )
    
    def __call__(self, x, mask=None):
        return x + self.ffn(self.ln2(x + self.attn(self.ln1(x), mask)))


class StandardGPT(nn.Module):
    """Standard GPT baseline"""
    def __init__(self, vocab: int, dim: int, layers: int, heads: int, mult: int):
        super().__init__()
        self.embed = StandardEmbed(vocab, dim)
        self.blocks = [StandardBlock(dim, heads, mult) for _ in range(layers)]
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
    
    def __call__(self, t, targets=None):
        x = self.embed(t)
        mask = mx.triu(mx.full((t.shape[1], t.shape[1]), -1e9), k=1)
        for b in self.blocks: x = b(x, mask)
        logits = self.head(self.ln_f(x))
        if targets is None: return logits
        loss = nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return loss, logits


def count_params(model):
    return sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))


def quick_train(model, name, train_files, val_tokens, steps=50):
    """Quick training run"""
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")
    
    opt = optim.Adam(learning_rate=0.001)
    n_params = count_params(model)
    print(f"Parameters: {n_params:,}")
    
    losses = []
    t0 = time.time()
    
    for step in range(steps):
        # Sample batch
        import random
        file = random.choice(train_files)
        tokens = np.fromfile(file, dtype=np.uint32).astype(np.int32)
        
        start = random.randint(0, max(0, len(tokens) - 1025))
        chunk = tokens[start:start+1025]
        if len(chunk) < 1025: continue
        
        x = mx.array(chunk[:-1].reshape(1, -1))
        y = mx.array(chunk[1:].reshape(1, -1))
        
        # Train
        def loss_fn(m):
            l, _ = m(x, y)
            return l
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        loss_val = float(loss.mean())
        losses.append(loss_val)
        
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        
        if step % 10 == 0:
            print(f"  Step {step:2d} | Loss: {sum(losses[-10:])/len(losses[-10:]):.4f}")
    
    elapsed = time.time() - t0
    
    # Evaluate
    print(f"\nEvaluating {name}...")
    model.eval()
    n_batches = len(val_tokens) // 1024
    total_loss = 0.0
    
    for i in range(min(n_batches, 50)):
        start = i * 1024
        x = mx.array(val_tokens[start:start+1024].reshape(1, -1))
        y = mx.array(val_tokens[start+1:start+1025].reshape(1, -1))
        loss, _ = model(x, y)
        total_loss += float(loss.item()) * 1024
    
    avg_loss = total_loss / (min(n_batches, 50) * 1024)
    bpb = avg_loss / math.log(2)
    
    print(f"  Val Loss: {avg_loss:.4f}")
    print(f"  Val BPB:  {bpb:.4f}")
    print(f"  Time:     {elapsed:.1f}s")
    
    model.train()
    
    return {
        'name': name,
        'params': n_params,
        'bpb': bpb,
        'loss': avg_loss,
        'time': elapsed
    }


def main():
    print("="*60)
    print("HYPERION vs Baseline - Quick Comparison")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_files = [f"./data/datasets/fineweb10B_sp1024/fineweb_train_{i:06d}.bin" 
                   for i in range(10)]
    val_file = "./data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"
    
    train_files = [f for f in train_files if __import__('os').path.exists(f)]
    val_tokens = __import__('numpy').fromfile(val_file, dtype=__import__('numpy').uint32).astype(__import__('numpy').int32)[:524288]
    
    print(f"✓ Found {len(train_files)} train files")
    print(f"✓ Val tokens: {len(val_tokens):,}")
    
    # HYPERION
    h_config = HyperionConfig()
    hyperion = Hyperion(h_config)
    h_results = quick_train(hyperion, "HYPERION (Sparse)", train_files, val_tokens)
    
    # Standard (matching dimensions for fair comparison)
    standard = StandardGPT(
        vocab=1024, 
        dim=64, 
        layers=8, 
        heads=8, 
        mult=3
    )
    s_results = quick_train(standard, "Standard (Dense)", train_files, val_tokens)
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    results = [h_results, s_results]
    
    print(f"\n{'Model':<20} {'Params':>12} {'BPB':>8} {'Size':>10}")
    print("-"*60)
    for r in results:
        size_mb = r['params'] * 4 / 1024 / 1024
        print(f"{r['name']:<20} {r['params']:>12,} {r['bpb']:>8.4f} {size_mb:>9.1f}MB")
    
    # Winner
    best = min(results, key=lambda x: x['bpb'])
    print(f"\n🏆 Best BPB: {best['name']} ({best['bpb']:.4f})")
    
    # Efficiency
    h_eff = h_results['bpb'] / (h_results['params'] / 1e6)
    s_eff = s_results['bpb'] / (s_results['params'] / 1e6)
    print(f"\nEfficiency (BPB per 1M params):")
    print(f"  HYPERION: {h_eff:.4f}")
    print(f"  Standard: {s_eff:.4f}")
    print(f"  Winner:   {'HYPERION' if h_eff < s_eff else 'Standard'}")
    
    # Compression
    compression = s_results['params'] / h_results['params']
    print(f"\nParameter reduction: {compression:.1f}x smaller")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
