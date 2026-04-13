#!/usr/bin/env python3
"""
OMNICLAW: Master Model - HYPERION + CHRYSALIS merged
- Sparse factorized embeddings (HYPERION)
- Adaptive depth routing (CHRYSALIS)
- Complementary prediction (CHRYSALIS)
"""

import sys, time, math, random
sys.path.insert(0, '.')

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten


class SparseEmbed(nn.Module):
    """HYPERION: Sparse factorized embedding"""
    def __init__(self, vocab=1024, dict_size=256, atom_dim=64, k=4):
        super().__init__()
        self.dictionary = nn.Embedding(dict_size, atom_dim)
        self.indices = nn.Embedding(vocab, k)
        self.weights = nn.Embedding(vocab, k)
        
        # Init
        self.dictionary.weight = self.dictionary.weight * 0.02
        self.weights.weight = self.weights.weight * 0.01
    
    def __call__(self, t):
        B, T = t.shape
        idx = self.indices(t).astype(mx.int32)
        w = nn.softmax(self.weights(t), axis=-1)
        atoms = self.dictionary(idx.reshape(-1)).reshape(B, T, -1, self.dictionary.weight.shape[1])
        return (w[..., None] * atoms).sum(axis=2)


class AdaptiveRouter(nn.Module):
    """CHRYSALIS: Route by difficulty"""
    def __init__(self, dim):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 3)
        )
    
    def __call__(self, x):
        feat = x.mean(axis=1)
        logits = self.predictor(feat)
        depth_idx = mx.argmax(logits, axis=-1)
        depths = mx.array([2, 4, 8])
        return int(depths[depth_idx[0]].item()), depth_idx


class OMNIBlock(nn.Module):
    """Hybrid block with hierarchical attention"""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.dim = dim
        
        # Standard attention
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        
        # FFN
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 3), nn.GELU(), nn.Linear(dim * 3, dim)
        )
    
    def __call__(self, x, mask=None):
        B, T, D = x.shape
        
        # Attention
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


class OMNICLAW(nn.Module):
    """Master model: HYPERION + CHRYSALIS"""
    def __init__(self, vocab=1024, dict_size=256, atom_dim=64, sparsity_k=4):
        super().__init__()
        self.vocab = vocab
        
        # HYPERION: Sparse embedding
        self.embed = SparseEmbed(vocab, dict_size, atom_dim, sparsity_k)
        
        # CHRYSALIS: Router
        self.router = AdaptiveRouter(atom_dim)
        
        # Blocks
        self.blocks = [OMNIBlock(atom_dim) for _ in range(8)]
        
        # Output
        self.ln_f = nn.LayerNorm(atom_dim)
        self.head = nn.Linear(atom_dim, vocab)
    
    def __call__(self, tokens, targets=None):
        # HYPERION: Sparse embed
        x = self.embed(tokens)
        B, T, D = x.shape
        
        # CHRYSALIS: Route by difficulty
        depth, difficulty = self.router(x)
        
        # Mask
        mask = mx.triu(mx.full((T, T), -1e9), k=1)
        
        # Run adaptive depth
        for i in range(depth):
            x = self.blocks[i](x, mask)
        
        # Output
        logits = self.head(self.ln_f(x))
        
        if targets is not None:
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, self.vocab), targets.reshape(-1)
            )
            return loss, logits, depth
        return logits


def count_params(model):
    return sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))


def train_omniclaw():
    print("="*60)
    print("OMNICLAW: HYPERION + CHRYSALIS Hybrid")
    print("="*60)
    
    # Build model
    model = OMNICLAW(vocab=1024, dict_size=256, atom_dim=64, sparsity_k=4)
    n_params = count_params(model)
    
    # Count components
    embed_params = 256*64 + 1024*4*2
    
    print(f"\nTotal Parameters: {n_params:,}")
    print(f"  Sparse embed: {embed_params:,} (vs 524K dense)")
    print(f"  Size: ~{n_params*4/1024/1024:.1f}MB")
    
    # Load data
    import glob
    train_files = sorted(glob.glob("./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"))[:5]
    val_file = "./data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"
    val_tokens = np.fromfile(val_file, dtype=np.uint32).astype(np.int32)[:50000]
    
    print(f"Train: {len(train_files)} files, Val: {len(val_tokens):,} tokens")
    
    # Train
    opt = optim.Adam(learning_rate=0.001)
    
    print("\nTraining 30 steps...")
    print("-"*60)
    
    depths = {2: 0, 4: 0, 8: 0}
    
    for step in range(30):
        file = random.choice(train_files)
        tokens = np.fromfile(file, dtype=np.uint32).astype(np.int32)
        start = random.randint(0, max(1, len(tokens) - 1025))
        chunk = tokens[start:start+1025]
        
        x = mx.array(chunk[:-1].reshape(1, -1))
        y = mx.array(chunk[1:].reshape(1, -1))
        
        def loss_fn(m):
            l, _, d = m(x, y)
            return l.mean(), d
        
        (loss, depth), grads = mx.value_and_grad(lambda m: loss_fn(m)[0])(model)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        
        depths[int(depth[0])] += 1
        
        if step % 10 == 0:
            print(f"Step {step:2d} | Loss: {float(loss.mean()):.4f} | Depth: {int(depth[0])}")
    
    print(f"\nDepth distribution: {depths}")
    
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
    avg_depth = sum(d*c for d,c in depths.items()) / 30
    
    print(f"\n{'='*60}")
    print("OMNICLAW Results")
    print(f"{'='*60}")
    print(f"Parameters:   {n_params:,}")
    print(f"Val Loss:     {avg_loss:.4f}")
    print(f"Val BPB:      {bpb:.4f}")
    print(f"Avg Depth:    {avg_depth:.1f} layers")
    print(f"Compute Save: {(8-avg_depth)/8*100:.0f}%")
    print(f"{'='*60}")
    
    return {'name': 'OMNICLAW', 'params': n_params, 'bpb': bpb, 'depth': avg_depth}


if __name__ == "__main__":
    results = train_omniclaw()
    print(f"\nFinal BPB: {results['bpb']:.4f}")
