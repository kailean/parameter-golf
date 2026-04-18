#!/usr/bin/env python3
"""Minimal test — does Kaggle give us a GPU?"""
import torch
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB", flush=True)
else:
    print("NO GPU AVAILABLE", flush=True)

# Quick compute test
if torch.cuda.is_available():
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x
    print(f"Compute test: PASS (result sum = {y.sum().item():.4f})", flush=True)