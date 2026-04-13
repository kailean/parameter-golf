"""
Q2: Compression-Aware Training — Byte Shuffle Permutations

Core insight: zstd compresses byte streams. The order of bytes within each tensor
determines how well zstd can find patterns. A learned permutation of bytes within
each weight tensor can dramatically improve compression ratio.

The permutation itself costs almost nothing: for a tensor of N rows, the permutation
is N integers (stored as uint16 if N < 65536), which is negligible.

The key trick: we can make this DIFFERENTIABLE by:
1. Sorting weight rows by their L2 norm (deterministic, zstd-friendly)
2. Learning a residual permutation on top (few params, big compression impact)
3. Adding a compression ratio proxy to the training loss

This is NOT the same as Procrustes rotation (which tried to minimize MSE before
compression and increased artifact size). We directly optimize for zstd ratio.

Implementation:
- During training: sort weight rows by norm, add compression proxy loss
- During serialization: apply learned shuffle before zstd compression
- During dequantization: apply inverse shuffle after dequantization
"""

import numpy as np
import mlx.core as mx
from collections import defaultdict


def compute_compression_proxy_loss(model, flat_state=None, sample_size=4):
    """
    Differentiable proxy for zstd compression ratio.
    
    Instead of running zstd (which is not differentiable), we compute:
    1. Row-wise entropy of each weight matrix (bytes per row after quantization)
    2. Correlation between adjacent rows (higher = better compression)
    3. Norm-based ordering consistency (sorted by norm = better compression)
    
    The loss penalizes high-entropy, low-correlation weight arrangements.
    This is a proxy — it won't be exact, but it pushes weights toward 
    zstd-friendly distributions.
    """
    if flat_state is None:
        flat_state = {k: v for k, v in mx.utils.tree_flatten(model.parameters())}
    
    total_penalty = mx.array(0.0)
    n_tensors = 0
    
    for name, arr in flat_state.items():
        if not mx.issubdtype(arr.dtype, mx.floating):
            continue
        if arr.size < 256:  # Skip tiny tensors
            continue
        
        # Reshape to 2D (rows × cols) — this is how int6 quantization works
        rows = arr.reshape(arr.shape[0], -1) if arr.ndim >= 2 else arr.reshape(1, -1)
        
        # 1. Norm-based ordering: rows sorted by norm compress better
        norms = mx.norm(rows, axis=1)  # (num_rows,)
        sorted_norms = mx.sort(norms)
        # Penalty for not being sorted by norm
        norm_disorder = mx.sum(mx.abs(norms - sorted_norms)) / (mx.sum(mx.abs(sorted_norms)) + 1e-8)
        
        # 2. Adjacent row correlation: higher correlation = better zstd
        if rows.shape[0] > 1:
            corr = mx.sum(rows[1:] * rows[:-1]) / (mx.norm(rows[1:]) * mx.norm(rows[:-1]) + 1e-8)
            corr_penalty = 1.0 - corr  # 0 = perfect correlation, 2 = anti-correlation
        else:
            corr_penalty = mx.array(0.0)
        
        total_penalty = total_penalty + norm_disorder + 0.1 * corr_penalty
        n_tensors += 1
        
        if n_tensors >= sample_size:
            break
    
    return total_penalty / max(n_tensors, 1)


def learn_byte_shuffle_permutation(weight_matrix: np.ndarray) -> np.ndarray:
    """
    Learn the optimal byte shuffle permutation for a weight matrix.
    
    Strategy: Sort rows by L2 norm, then within each norm-group, sort by
    the first principal component direction. This maximizes zstd's ability
    to find patterns in the byte stream.
    
    Returns: permutation array (new_index -> original_index)
    """
    if weight_matrix.ndim < 2:
        return np.arange(weight_matrix.shape[0])
    
    # Reshape to 2D
    rows = weight_matrix.reshape(weight_matrix.shape[0], -1)
    
    # Sort by L2 norm (primary)
    norms = np.linalg.norm(rows, axis=1)
    norm_order = np.argsort(norms)
    
    # Within norm groups, sort by first PC direction (secondary)
    # This clusters similar rows together for better zstd
    if rows.shape[0] > 32:
        try:
            # Compute covariance and first PC
            centered = rows[norm_order] - rows[norm_order].mean(axis=0, keepdims=True)
            cov = centered.T @ centered / centered.shape[0]
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            pc1 = eigenvectors[:, -1]  # First principal component
            projections = rows @ pc1
            
            # Stable sort: norm first, then PC projection within norm groups
            # Group rows into norm buckets
            n_groups = min(16, rows.shape[0] // 4)
            norm_ranks = np.argsort(np.argsort(norms)) * n_groups // rows.shape[0]
            
            # Secondary sort within each norm group
            final_order = []
            for g in range(n_groups):
                mask = norm_ranks == g
                group_indices = np.where(mask)[0]
                group_projections = projections[group_indices]
                sub_order = np.argsort(group_projections)
                final_order.extend(group_indices[sub_order].tolist())
            
            if len(final_order) == rows.shape[0]:
                return np.array(final_order)
        except Exception:
            pass
    
    return norm_order


def apply_shuffle_to_quantized(quant_obj, model_state=None):
    """
    Apply learned byte shuffles to a quantized state dict before compression.
    
    For each quantized weight matrix:
    1. Compute optimal row permutation (sorted by norm + PC)
    2. Apply permutation to the packed int6 data
    3. Store the permutation in the quant object (negligible size)
    
    This is applied AFTER quantization but BEFORE zstd compression.
    """
    import copy
    shuffled = copy.deepcopy(quant_obj)
    
    permutations = {}
    shuffles_key = "__shuffles__"
    
    for name in list(quant_obj.get("quantized", {}).keys()):
        packed = quant_obj["quantized"][name]
        if not isinstance(packed, np.ndarray) or packed.ndim < 1:
            continue
        
        # For int6 packed per row: shape is (num_rows, packed_cols)
        if packed.ndim == 2 and packed.shape[0] > 1:
            # Compute permutation from the scale matrix (proxy for row norms)
            scales = quant_obj["scales"].get(name)
            if scales is not None and scales.ndim >= 1:
                # Sort rows by scale magnitude (scale = row norm proxy)
                row_norms = np.abs(scales.flatten()[:packed.shape[0]])
                perm = np.argsort(row_norms)
                permutations[name] = perm
                
                # Apply permutation
                packed_shuffled = packed[perm]
                quant_obj["quantized"][name] = packed_shuffled
                
                if scales.ndim == 1:
                    quant_obj["scales"][name] = scales[perm]
                elif scales.ndim == 2:
                    quant_obj["scales"][name] = scales[perm]
    
    # Store permutations (they're tiny — just uint16 indices)
    if permutations:
        max_perm_len = max(len(p) for p in permutations.values())
        if max_perm_len < 65536:
            perm_dtype = np.uint16
        else:
            perm_dtype = np.uint32
        
        shuffled_permutations = {}
        for name, perm in permutations.items():
            shuffled_permutations[name] = perm.astype(perm_dtype)
        
        shuffled[shuffles_key] = shuffled_permutations
    
    return shuffled


def apply_inverse_shuffle(dequant_state, shuffles):
    """
    Apply inverse shuffle to restore original row order after dequantization.
    """
    if shuffles is None:
        return dequant_state
    
    for name, perm in shuffles.items():
        if name in dequant_state:
            arr = dequant_state[name]
            if arr.ndim >= 2:
                # Inverse permutation
                inv_perm = np.argsort(perm)
                dequant_state[name] = arr[inv_perm]
    
    return dequant_state


def measure_compression_savings(flat_state, args=None):
    """
    Measure the compression ratio with and without byte shuffling.
    Returns (baseline_bytes, shuffled_bytes, savings_pct).
    """
    import zstandard
    
    # Baseline: quantize and compress without shuffle
    from train_gpt_mlx_kl import quantize_state_dict_int6
    quant_obj, stats = quantize_state_dict_int6(flat_state, args)
    import pickle
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    baseline_bytes = len(zstandard.ZstdCompressor(level=22).compress(quant_raw))
    
    # With shuffle
    shuffled_obj = apply_shuffle_to_quantized(quant_obj)
    shuffled_raw = pickle.dumps(shuffled_obj, protocol=pickle.HIGHEST_PROTOCOL)
    shuffled_bytes = len(zstandard.ZstdCompressor(level=22).compress(shuffled_raw))
    
    savings_pct = (baseline_bytes - shuffled_bytes) / baseline_bytes * 100
    return baseline_bytes, shuffled_bytes, savings_pct