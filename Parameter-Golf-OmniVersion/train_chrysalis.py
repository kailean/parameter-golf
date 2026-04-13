#!/usr/bin/env python3
"""
CHRYSALIS: Adaptive Depth + Runtime Knowledge System
Phase 1: Adaptive computation routing
"""

import sys, time, math, random
sys.path.insert(0, '.')

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten


class AdaptiveRouter(nn.Module):
    """CHRYSALIS: Route tokens by predicted difficulty"""
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)  # easy/medium/hard
        )
        self.depths = [2, 4, 8]  # shallow/medium/deep
    
    def __call__(self, x):
        """x: (batch, seq, dim) -> depths: (batch,)"""
        # Predict from mean embedding
        feat = x.mean(axis=1)  # (batch, dim)
        logits = self.predictor(feat)
        difficulty = mx.argmax(logits, axis=-1)
        return self.depths[difficulty[0]], difficulty


class AdaptiveBlock(nn.Module):
    """Transformer block that can run at different depths"""
    def __init__(self, dim, heads=8, mult=3):
        super().__init__()
        self.dim = dim
        
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )
    
    def __call__(self, x, mask=None):
        # Attention
        B, T, D = x.shape
        qkv = self.qkv(self.ln1(x)).reshape(B, T, 3, D)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        
        scores = (q @ k.transpose(0, 2, 1)) / math.sqrt(D)
        if mask is not None:
            scores = scores + mask
        attn = nn.softmax(scores, axis=-1) @ v
        x = x + self.out(attn)
        
        # FFN
        x = x + self.ffn(self.ln2(x))
        return x


class CHRYSALIS(nn.Module):
    """CHRYSALIS with adaptive depth routing"""
    def __init__(self, vocab=1024, dim=512, layers=8, heads=8):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.max_layers = layers
        
        # Standard embedding
        self.embed = nn.Embedding(vocab, dim)
        
        # Adaptive router
        self.router = AdaptiveRouter(dim)
        
        # Blocks (shared weights for efficiency)
        self.blocks = [AdaptiveBlock(dim, heads) for _ in range(layers)]
        
        # Output
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
    
    def __call__(self, tokens, targets=None):
        x = self.embed(tokens)
        B, T, D = x.shape
        
        # Route by difficulty
        depth, difficulty = self.router(x)
        
        # Causal mask
        mask = mx.triu(mx.full((T, T), -1e9), k=1)
        
        # Run adaptive depth
        for i in range(depth):
            x = self.blocks[i](x, mask)
        
        # Output
        logits = self.head(self.ln_f(x))
        
        if targets is not None:
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, self.vocab),
                targets.reshape(-1)
            )
            return loss, logits, depth
        
        return logits


def count_params(model):
    return sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))


def train_chrysalis():
    print("="*60)
    print("CHRYSALIS: Adaptive Depth Training")
    print("="*60)
    
    # Build model
    model = CHRYSALIS(vocab=1024, dim=512, layers=8)
    n_params = count_params(model)
    print(f"\nParameters: {n_params:,} (~{n_params*4/1024/1024:.1f}MB)")
    
    # Load data
    import glob
    train_files = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"))[:5]
    val_file = "./data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"
    val_tokens = np.fromfile(val_file, dtype=np.uint32).astype(np.int32)[:50000]
    
    print(f"Train: {len(train_files)} files, Val: {len(val_tokens):,} tokens")
    
    # Optimizer
    opt = optim.Adam(learning_rate=0.001)
    
    # Train
    print("\nTraining 30 steps...")
    print("-"*60)
    
    depths_used = {2: 0, 4: 0, 8: 0}
    
    for step in range(30):
        # Sample batch
        file = random.choice(train_files)
        tokens = np.fromfile(file, dtype=np.uint32).astype(np.int32)
        start = random.randint(0, max(1, len(tokens) - 1025))
        chunk = tokens[start:start+1025]
        
        x = mx.array(chunk[:-1].reshape(1, -1))
        y = mx.array(chunk[1:].reshape(1, -1))
        
        # Forward
        def loss_fn(m):
            l, _, d = m(x, y)
            return l.mean()
        
        loss, grads = mx.value_and_grad(loss_fn)(model)
        # Get depth separately (no grad needed)
        _, _, depth = model(x, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        
        depths_used[int(depth[0])] += 1
        
        if step % 10 == 0:
            print(f"Step {step:2d} | Loss: {float(loss.mean()):.4f} | Depth: {int(depth[0])}")
    
    # Stats
    print(f"\nDepth distribution:")
    for d, c in depths_used.items():
        print(f"  Depth {d}: {c} steps ({100*c/30:.0f}%)")
    
    # Evaluate
    print("\nEvaluating...")
    model.eval()
    total_loss = 0.0
    
    for i in range(20):
        start = i * 1024
        x = mx.array(val_tokens[start:start+1024].reshape(1, -1))
        y = mx.array(val_tokens[start+1:start+1025].reshape(1, -1))
        
        loss, _, _ = model(x, y)
        total_loss += float(loss.mean()) * 1024
    
    avg_loss = total_loss / (20 * 1024)
    bpb = avg_loss / math.log(2)
    
    print(f"\n{'='*60}")
    print("CHRYSALIS Results")
    print(f"{'='*60}")
    print(f"Parameters: {n_params:,}")
    print(f"Val Loss:   {avg_loss:.4f}")
    print(f"Val BPB:    {bpb:.4f}")
    print(f"Avg Depth:  {sum(d*c for d,c in depths_used.items())/30:.1f} layers")
    print(f"{'='*60}")
    
    return {'name': 'CHRYSALIS', 'params': n_params, 'bpb': bpb}


if __name__ == "__main__":
    results = train_chrysalis()
    print(f"\nFinal BPB: {results['bpb']:.4f}")
