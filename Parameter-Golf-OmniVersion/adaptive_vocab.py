"""
Q3: Adaptive Token Vocabulary — The Hidden Variable

The SP1024 tokenizer is a fixed 1024-token BPE vocabulary. But BPB = loss / ln(2) × (tokens/bytes).
A suboptimal tokenizer means:
1. More tokens per byte (higher tokenization ratio)
2. Higher per-token loss (rare tokens aren't well-learned)

Everyone accepts the tokenizer as fixed. But we can:
1. ANALYZE the tokenizer to find inefficiencies
2. LEARN which tokens the model predicts well vs poorly
3. Use the BackoffNgramMixer to handle poorly-predicted tokens

Key insight: If we can identify a subset of tokens that the n-gram mixer
handles perfectly, the neural model only needs to predict the remaining tokens.
This is the "vocabulary bypass" — the mixer acts as a learned tokenizer extension.

This module provides:
- Token efficiency analysis (bytes/token distribution)
- Per-token loss analysis (which tokens waste model capacity)
- Adaptive n-gram bypass (skip neural model for high-confidence tokens)
"""

import numpy as np
from collections import defaultdict


def analyze_tokenizer_efficiency(sp_model_path: str):
    """
    Analyze the SP1024 tokenizer to find efficiency gaps.
    
    Returns statistics about:
    - Bytes per token distribution (mean, std, min, max)
    - Token frequency distribution (how many tokens are rarely used)
    - Potential bypass candidates (tokens the n-gram mixer can handle)
    """
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    
    vocab_size = sp.vocab_size()
    
    # Compute bytes per token
    bytes_per_token = []
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        try:
            byte_count = len(piece.encode('utf-8'))
        except:
            byte_count = 1
        bytes_per_token.append(max(1, byte_count))
    
    bytes_per_token = np.array(bytes_per_token)
    
    stats = {
        'vocab_size': vocab_size,
        'mean_bytes_per_token': float(bytes_per_token.mean()),
        'std_bytes_per_token': float(bytes_per_token.std()),
        'min_bytes_per_token': int(bytes_per_token.min()),
        'max_bytes_per_token': int(bytes_per_token.max()),
        'tokens_with_1_byte': int((bytes_per_token == 1).sum()),
        'tokens_with_2_bytes': int((bytes_per_token == 2).sum()),
        'tokens_with_3_bytes': int((bytes_per_token == 3).sum()),
        'tokens_with_4_bytes': int((bytes_per_token >= 4).sum()),
    }
    
    return stats


def analyze_per_token_loss(token_losses: np.ndarray, token_ids: np.ndarray, 
                           vocab_size: int = 1024, top_k: int = 20):
    """
    Analyze which tokens contribute most to BPB.
    
    Args:
        token_losses: (N,) array of per-token losses
        token_ids: (N,) array of token IDs
        vocab_size: vocabulary size
        top_k: number of worst tokens to report
    
    Returns:
        Dict with per-token loss statistics and bypass candidates
    """
    # Accumulate loss per token
    total_loss = defaultdict(float)
    count = defaultdict(int)
    
    for loss, tid in zip(token_losses, token_ids):
        total_loss[int(tid)] += float(loss)
        count[int(tid)] += 1
    
    # Compute average loss per token
    avg_loss = {}
    for tid in total_loss:
        avg_loss[tid] = total_loss[tid] / count[tid]
    
    # Sort by average loss (highest = worst predicted)
    sorted_tokens = sorted(avg_loss.items(), key=lambda x: x[1], reverse=True)
    
    # Compute cumulative BPB contribution
    total_bpb = sum(total_loss.values())
    
    # Identify bypass candidates: tokens where bigram context alone predicts well
    # These are tokens with high frequency AND moderate loss (the n-gram can handle them)
    bypass_candidates = []
    for tid, loss in sorted_tokens:
        freq = count[tid]
        if freq > 100 and loss < 3.0:  # Frequent, moderate loss
            bypass_candidates.append({
                'token_id': tid,
                'avg_loss': loss,
                'frequency': freq,
                'bpb_contribution': total_loss[tid] / total_bpb,
            })
    
    return {
        'worst_tokens': sorted_tokens[:top_k],
        'best_tokens': sorted_tokens[-top_k:],
        'bypass_candidates': bypass_candidates[:top_k],
        'total_tokens': len(token_ids),
        'unique_tokens': len(count),
        'total_loss': total_bpb,
        'avg_loss': total_bpb / len(token_ids),
    }


class AdaptiveNgramBypass:
    """
    Eval-time bypass: skip the neural model for tokens that n-grams predict
    with high confidence.
    
    This is different from BackoffNgramMixer (which always runs both and mixes).
    The bypass skips neural model computation entirely for high-confidence n-gram
    predictions, saving eval time AND improving accuracy on easy tokens.
    
    Threshold: if max n-gram probability > bypass_threshold, use n-gram directly.
    Otherwise, fall through to neural model + n-gram mixing.
    """
    
    def __init__(self, vocab_size=1024, max_order=7, bypass_threshold=0.95,
                 hash_buckets=2_000_000):
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.bypass_threshold = bypass_threshold
        self.hash_buckets = hash_buckets
        self._counts = [
            defaultdict(lambda: np.zeros(vocab_size, dtype=np.float32))
            for _ in range(max_order + 1)
        ]
        self._total = [defaultdict(float) for _ in range(max_order + 1)]
    
    def _hash_ctx(self, context_tokens):
        h = 0
        for t in context_tokens:
            h = (h * 31337 + int(t)) % self.hash_buckets
        return h
    
    def _ngram_probs(self, context_tokens):
        """Get n-gram probability distribution."""
        import math
        V = self.vocab_size
        probs = np.ones(V, dtype=np.float64) / V
        for order in range(1, self.max_order + 1):
            if len(context_tokens) < order:
                break
            ctx_hash = self._hash_ctx(context_tokens[-order:])
            total = self._total[order][ctx_hash]
            if total <= 0:
                continue
            lam = total / (total + 5.0)
            c = self._counts[order][ctx_hash].astype(np.float64)
            order_probs = (c + 1e-10) / (total + 1e-10 * V)
            order_probs /= order_probs.sum()
            probs = (1.0 - lam) * probs + lam * order_probs
        probs /= probs.sum()
        return probs
    
    def should_bypass(self, context_tokens) -> bool:
        """
        Check if n-gram prediction is confident enough to bypass the neural model.
        
        Returns True if max P(token | context) > bypass_threshold.
        """
        probs = self._ngram_probs(context_tokens)
        max_prob = probs.max()
        return max_prob > self.bypass_threshold
    
    def get_bypass_prediction(self, context_tokens):
        """Get the n-gram prediction (call only if should_bypass returned True)."""
        probs = self._ngram_probs(context_tokens)
        return int(np.argmax(probs)), float(probs.max())
    
    def update(self, context_tokens, token_id: int):
        """Update n-gram counts with observed token."""
        for order in range(1, self.max_order + 1):
            if len(context_tokens) >= order:
                ctx_hash = self._hash_ctx(context_tokens[-order:])
                self._counts[order][ctx_hash][token_id] += 1.0
                self._total[order][ctx_hash] += 1.0


def compute_bytes_per_token_lut(sp_model_path: str):
    """
    Pre-compute bytes-per-token lookup table for BPB-aware training.
    
    This is the implementation for Q3/Angle 1 from pg_novel_ideas.md:
    Byte-weighted cross-entropy loss that weights tokens by their UTF-8 byte count.
    
    Tokens decoding to more bytes get proportionally more gradient signal,
    which directly optimizes for BPB (bits per byte) rather than CE (bits per token).
    """
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)
    
    vocab_size = sp.vocab_size()
    bytes_lut = np.ones(vocab_size, dtype=np.float32)
    
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        try:
            byte_count = len(piece.encode('utf-8'))
        except:
            byte_count = 1
        bytes_lut[i] = max(1, byte_count)
    
    return bytes_lut


if __name__ == "__main__":
    # Quick analysis
    import os
    tokenizer_path = "./data/tokenizers/fineweb_1024_bpe.model"
    if os.path.exists(tokenizer_path):
        stats = analyze_tokenizer_efficiency(tokenizer_path)
        print("Tokenizer Efficiency Analysis:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    else:
        print(f"Tokenizer not found at {tokenizer_path}")
        print("Run: python data/cached_challenge_fineweb.py --variant sp1024 to download")