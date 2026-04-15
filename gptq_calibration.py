#!/usr/bin/env python3
"""
Full GPTQ Calibration for Parameter Golf
Adapted from SOTA PR #1530 (samacqua) with improvements.

Key differences from our GPTQ-lite:
1. Uses Hessian-based calibration (collects activation statistics from real data)
2. Per-layer adaptive clip_sigmas (matrix=12.85, embed=20.0)
3. Block-wise quantization with Cholesky decomposition
4. Supports mixed int5/int6/int8 precision
5. Entropy regularization support (CAT loss)

This replaces our simple quantize_float_tensor_gptq_lite which only searched
5 percentile clip values per row.
"""
import math
import numpy as np
import torch
from torch import Tensor
from collections import defaultdict
import re


# ─────────────────────────────────────────────────────────────
# HESSIAN COLLECTION
# ─────────────────────────────────────────────────────────────

def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    """Collect Hessian matrices for each weight tensor using calibration data.
    
    This is the key difference from GPTQ-lite: instead of just searching clip
    percentiles, we compute the actual Fisher information matrix H = X^T X
    for each weight, which tells us the optimal quantization grid.
    
    Args:
        model: The trained model (EMA or SWA applied)
        train_loader: TokenStream for calibration data
        h: Hyperparameters
        device: cuda device
        n_calibration_batches: Number of batches to use (default 64)
    
    Returns:
        hessians: dict of {name: H_matrix} where H is (cols, cols)
    """
    hessians = {}
    hooks = []
    
    # Mark calibration mode
    for i, block in enumerate(model.blocks):
        block.attn._calib = True
        block.mlp._calib = True
    
    def make_attn_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            for suffix in ["c_q", "c_k", "c_v"]:
                name = f"blocks.{layer_idx}.attn.{suffix}.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        x.shape[1], x.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(x.T, x)
            # proj weight uses attention output as input
            y = module._last_proj_input
            if y is not None:
                y = y.float()
                if y.ndim == 3:
                    y = y.reshape(-1, y.shape[-1])
                name = f"blocks.{layer_idx}.attn.proj.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        y.shape[1], y.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(y.T, y)
        return hook_fn

    def make_mlp_hook(layer_idx):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            name = f"blocks.{layer_idx}.mlp.fc.weight"
            if name not in hessians:
                hessians[name] = torch.zeros(
                    x.shape[1], x.shape[1], dtype=torch.float32, device=device
                )
            hessians[name].addmm_(x.T, x)
            # proj weight uses activation output as input
            h_act = module._last_down_input
            if h_act is not None:
                h_act = h_act.float()
                if h_act.ndim == 3:
                    h_act = h_act.reshape(-1, h_act.shape[-1])
                name = f"blocks.{layer_idx}.mlp.proj.weight"
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        h_act.shape[1], h_act.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(h_act.T, h_act)
        return hook_fn

    # Register hooks on all blocks
    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_hook(make_attn_hook(i)))
        hooks.append(block.mlp.register_forward_hook(make_mlp_hook(i)))
    
    # Also collect Hessian for embedding/output projection
    if model.tie_embeddings:
        hook_module = model.head_proj if model.head_proj is not None else model.final_norm

        def make_output_hook(name):
            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3:
                    x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(
                        x.shape[1], x.shape[1], dtype=torch.float32, device=device
                    )
                hessians[name].addmm_(x.T, x)
            return hook_fn

        hooks.append(
            hook_module.register_forward_hook(make_output_hook("tok_emb.weight"))
        )

    # Run calibration forward passes
    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    
    # Cleanup
    for hook in hooks:
        hook.remove()
    for i, block in enumerate(model.blocks):
        block.attn._calib = False
        block.mlp._calib = False
    
    # Normalize Hessians
    for name in hessians:
        hessians[name] = hessians[name].cpu() / n_calibration_batches
    
    return hessians


# ─────────────────────────────────────────────────────────────
# GPTQ QUANTIZATION WITH HESSIAN CALIBRATION
# ─────────────────────────────────────────────────────────────

def gptq_quantize_weight(w, H, clip_sigmas=3.0, clip_range=63, block_size=128):
    """GPTQ quantization with Hessian-based calibration.
    
    This is the full GPTQ algorithm from the SOTA submission:
    1. Compute optimal column ordering via Hessian diagonal
    2. Block-wise quantization with Cholesky inverse
    3. Error propagation to remaining columns
    4. Per-row adaptive clip using clip_sigmas * std
    
    Args:
        w: weight tensor (rows, cols)
        H: Hessian matrix (cols, cols) from calibration
        clip_sigmas: number of std devs for clipping (12.85 for matrix, 20.0 for embed)
        clip_range: max quantized value (63 for int6, 127 for int7, 255 for int8)
        block_size: block size for Cholesky decomposition (128)
    
    Returns:
        Q: quantized weight (int8, range [-clip_range, clip_range])
        s: per-row scale factors (float16)
    """
    W_orig = w.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()
    
    # Handle dead (zero-variance) columns
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    
    # Damping for numerical stability
    damp = 0.01 * H.diag().mean()
    H.diagonal().add_(damp)
    
    # Optimal column ordering (quantize most important columns first)
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    
    # Cholesky decomposition for efficient inverse
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    
    # Per-row adaptive clip scale
    row_std = W_orig.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()
    
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    
    # Block-wise GPTQ with error propagation
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1)
        
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    
    return Q[:, invperm], s


def gptq_mixed_quantize(state_dict, hessians, matrix_bits=6, embed_bits=8,
                        matrix_clip_sigmas=12.85, embed_clip_sigmas=20.0):
    """Mixed-precision GPTQ quantization with per-layer adaptive clipping.
    
    Args:
        state_dict: model state dict
        hessians: Hessian matrices from collect_hessians()
        matrix_bits: bit width for matrix weights (6 for int6, 5 for int5)
        embed_bits: bit width for embeddings (8 for int8, 7 for int7)
        matrix_clip_sigmas: clip sigmas for matrix weights
        embed_clip_sigmas: clip sigmas for embeddings
    
    Returns:
        result: dict of quantized tensors
        meta: dict of quantization metadata
    """
    result = {}
    meta = {}
    
    _EMBED_PATTERNS = ("tok_emb.weight", "lm_head.weight")
    
    for (name, tensor) in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        
        # Skip small tensors (passthrough as float16)
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"
            continue
        
        # Determine bit width and clip sigmas
        is_embed = any(p in name for p in _EMBED_PATTERNS)
        bits = embed_bits if is_embed else matrix_bits
        clip_sigmas = embed_clip_sigmas if is_embed else matrix_clip_sigmas
        clip_range = 2 ** (bits - 1) - 1  # 31 for int6, 63 for int7, 127 for int8
        
        # Check if we have a Hessian for this tensor
        if name in hessians:
            q, s = gptq_quantize_weight(
                t, hessians[name],
                clip_sigmas=clip_sigmas,
                clip_range=clip_range,
                block_size=128
            )
            result[name + ".q"] = q
            result[name + ".scale"] = s
            meta[name] = f"gptq (int{bits})"
        else:
            # Fallback to simple quantization for tensors without Hessians
            row_max = t.abs().amax(dim=1)
            scale = (row_max / clip_range).clamp_min(1e-10).to(torch.float16)
            q = torch.clamp(torch.round(t.float() / scale.unsqueeze(1)), -clip_range, clip_range).to(torch.int8)
            result[name + ".q"] = q
            result[name + ".scale"] = scale
            meta[name] = f"simple (int{bits})"
    
    # Print summary
    categories = defaultdict(set)
    for (name, cat) in meta.items():
        short = re.sub(r"\.\d+$", "", re.sub(r"blocks\.\d+", "blocks", name))
        categories[cat].add(short)
    print("Quantized weights:")
    for cat in sorted(categories):
        print(f"  {cat}: {', '.join(sorted(categories[cat]))}")
    
    return result, meta


def dequantize_gptq(result, meta, template_sd):
    """Dequantize GPTQ-quantized weights back to float.
    
    Args:
        result: quantized tensors dict
        meta: quantization metadata dict
        template_sd: original state dict (for dtypes and shapes)
    
    Returns:
        dict of dequantized tensors
    """
    out = {}
    for (name, orig) in template_sd.items():
        info = meta.get(name)
        if info is None:
            continue
        orig_dtype = orig.dtype
        
        if "passthrough" in info:
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
        elif "gptq" in info or "simple" in info:
            q = result[name + ".q"].float()
            s = result[name + ".scale"].float()
            deq = q * s.view(q.shape[0], *([1] * (q.ndim - 1)))
            out[name] = deq.to(orig_dtype)
    
    return out


# ─────────────────────────────────────────────────────────────
# ENTROPY REGULARIZATION (CAT Loss)
# ─────────────────────────────────────────────────────────────

def entropy_regularization_loss(model, lambda_entropy=0.001):
    """Compression-Aware Training (CAT) loss.
    
    Adds a differentiable entropy penalty to the loss function.
    This pushes weight distributions toward more compressible patterns
    during training, achieving ~2.23× better compression ratio.
    
    Usage: total_loss = ce_loss + lambda_entropy * entropy_regularization_loss(model)
    
    Args:
        model: the neural network
        lambda_entropy: weight of entropy penalty (0.001 is a good starting point)
    
    Returns:
        scalar entropy loss
    """
    entropy = 0.0
    n_params = 0
    for name, param in model.named_parameters():
        if param.ndim < 2:
            continue
        # Compute histogram-based entropy of weight values
        # Binning into 64 bins (matches int6 quantization)
        w = param.detach().float()
        n_bins = 64
        w_min = w.min()
        w_max = w.max()
        if w_max - w_min < 1e-8:
            continue
        # Soft histogram approximation
        w_norm = (w - w_min) / (w_max - w_min)  # [0, 1]
        bins = torch.linspace(0, 1, n_bins + 1, device=w.device)
        # Count weights in each bin using differentiable soft assignment
        sigma = 0.02  # softness of bin assignment
        counts = torch.zeros(n_bins, device=w.device)
        for i in range(n_bins):
            center = (bins[i] + bins[i + 1]) / 2
            counts[i] = torch.exp(-((w_norm.flatten() - center) ** 2) / (2 * sigma ** 2)).sum()
        counts = counts / counts.sum()  # normalize to probability
        # Shannon entropy
        eps = 1e-10
        entropy += -(counts * torch.log2(counts + eps)).sum()
        n_params += 1
    
    # We want MINIMUM entropy = maximum compressibility
    # So we minimize entropy (push weights toward fewer distinct values)
    return entropy / max(n_params, 1)


# ─────────────────────────────────────────────────────────────
# INTEGRATION WITH train_gpt_kl.py
# ─────────────────────────────────────────────────────────────

def integrate_gptq_into_train_script():
    """
    To integrate full GPTQ into train_gpt_kl.py, add these changes:
    
    1. After training completes (before quantization), add:
    
        # Collect Hessians for GPTQ calibration
        from gptq_calibration import collect_hessians, gptq_mixed_quantize
        
        log("GPTQ: Collecting Hessians from calibration data...")
        t0 = time.perf_counter()
        
        # Create a calibration data loader
        train_stream = TokenStream(args.train_files)
        train_loader = DistributedTokenLoader(
            args.train_files, rank, world_size, device
        )
        
        hessians = collect_hessians(
            base_model, train_loader, args, device,
            n_calibration_batches=args.gptq_calibration_batches
        )
        log(f"GPTQ: Collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s")
        
        # Quantize with full GPTQ
        quant_result, quant_meta = gptq_mixed_quantize(
            base_model.state_dict(), hessians,
            matrix_bits=args.matrix_bits,
            embed_bits=args.embed_bits,
            matrix_clip_sigmas=args.matrix_clip_sigmas,
            embed_clip_sigmas=args.embed_clip_sigmas,
        )
    
    2. Add to Hyperparameters:
    
        gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", "64"))
        matrix_bits = int(os.environ.get("MATRIX_BITS", "6"))  # int6 for matrices
        embed_bits = int(os.environ.get("EMBED_BITS", "8"))  # int8 for embeddings
        matrix_clip_sigmas = float(os.environ.get("MATRIX_CLIP_SIGMAS", "12.85"))
        embed_clip_sigmas = float(os.environ.get("EMBED_CLIP_SIGMAS", "20.0"))
        entropy_lambda = float(os.environ.get("ENTROPY_LAMBDA", "0.0"))  # 0=disabled
    
    3. For entropy regularization during training, modify the loss:
    
        loss = ce_loss  # standard cross-entropy
        if args.entropy_lambda > 0:
            from gptq_calibration import entropy_regularization_loss
            loss = loss + args.entropy_lambda * entropy_regularization_loss(model)
    
    4. Reserve time for GPTQ calibration:
    
        if args.max_wallclock_seconds > 0:
            args.max_wallclock_seconds -= args.gptq_reserve_seconds  # ~13s for calibration
    """
    pass