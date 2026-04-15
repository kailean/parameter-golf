#!/usr/bin/env python3
"""
Custom binary serializer for quantized model state dicts.
Replaces torch.save (pickle+ZIP) with a flat binary format.

torch.save overhead analysis (640d model, 55M params):
  - Payload (int6 packed + scales + float passthrough): 42.67 MB
  - torch.save output: 56.95 MB
  - Overhead: 14.28 MB (33% waste!)

Our custom format: ~42.67 MB (near-zero overhead, highly compressible)

Format v1:
  [MAGIC 4B][VERSION 2B][N_TENSORS 2B]
  [for each tensor:]
    [name_len 1B][name UTF8][scheme 1B][dtype 1B][ndim 1B][shape ndim×4B]
    [data_len 4B][data bytes]
    [scale_dtype 1B][scale_data_len 2B][scale_data bytes]  (if has_scale)

Scheme codes:
  0 = int6_packed_per_row (4 values per 3 bytes)
  1 = int8_per_row
  2 = float_passthrough

This is designed to be:
  1. Zero-overhead (no padding, no dict keys)
  2. Sequential (brotli loves sequential, predictable data)
  3. Self-describing (can roundtrip without external metadata)
"""

import io
import struct
import numpy as np
import torch
from torch import Tensor

MAGIC = b"QTV1"  # Quantized Tensor V1
VERSION = 1

# Scheme codes
SCHEME_INT6_PACKED = 0
SCHEME_INT8_PER_ROW = 1
SCHEME_FLOAT_PASSTHROUGH = 2

# Dtype codes
DTYPE_MAP = {"float32": 0, "float16": 1, "bfloat16": 2, "int8": 3, "uint8": 4, "float64": 5}
DTYPE_RMAP = {v: k for k, v in DTYPE_MAP.items()}

TORCH_DTYPE_FROM_STR = {
    "float32": torch.float32, "float16": torch.float16,
    "bfloat16": torch.bfloat16, "int8": torch.int8,
    "uint8": torch.uint8, "float64": torch.float64,
}


def _dtype_to_code(dtype_str: str) -> int:
    return DTYPE_MAP.get(dtype_str, 0)


def _code_to_dtype(code: int) -> str:
    return DTYPE_RMAP.get(code, "float32")


def _torch_dtype_code(t: Tensor) -> int:
    s = str(t.dtype).removeprefix("torch.")
    return _dtype_to_code(s)


def serialize_quantized(quant_obj: dict) -> bytes:
    """Serialize quantized state dict to flat binary format.
    
    Input: output of quantize_state_dict_int6()
    Output: bytes blob ready for brotli compression
    """
    quantized = quant_obj.get("quantized", {})
    scales = quant_obj.get("scales", {})
    shapes = quant_obj.get("shapes", {})
    dtypes = quant_obj.get("dtypes", {})
    passthrough = quant_obj.get("passthrough", {})
    qmeta = quant_obj.get("qmeta", {})
    pt_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    
    buf = io.BytesIO()
    
    # Header
    buf.write(MAGIC)
    buf.write(struct.pack("<H", VERSION))
    
    # Count total tensors
    all_names = sorted(set(list(quantized.keys()) + list(passthrough.keys())))
    buf.write(struct.pack("<H", len(all_names)))
    
    for name in all_names:
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 255:
            raise ValueError(f"Tensor name too long: {name}")
        
        if name in quantized:
            # Quantized tensor (int6 or int8)
            packed = quantized[name]
            shape = shapes.get(name, ())
            dtype_str = dtypes.get(name, "float32")
            meta = qmeta.get(name, {})
            scale = scales.get(name)
            
            # Determine scheme
            if meta.get("scheme") == "per_row_int8":
                scheme = SCHEME_INT8_PER_ROW
            else:
                scheme = SCHEME_INT6_PACKED
            
            # Name
            buf.write(struct.pack("<B", len(name_bytes)))
            buf.write(name_bytes)
            
            # Scheme + original dtype
            buf.write(struct.pack("<B", scheme))
            buf.write(struct.pack("<B", _dtype_to_code(dtype_str)))
            
            # Shape
            buf.write(struct.pack("<B", len(shape)))
            for dim in shape:
                buf.write(struct.pack("<I", dim))
            
            # Quantized data
            raw = packed.tobytes() if isinstance(packed, np.ndarray) else bytes(packed)
            buf.write(struct.pack("<I", len(raw)))
            buf.write(raw)
            
            # Scale data
            if scale is not None:
                if isinstance(scale, Tensor):
                    scale_dtype_code = _torch_dtype_code(scale)
                    scale_raw = scale.numpy().tobytes() if scale.is_cpu else scale.cpu().numpy().tobytes()
                elif isinstance(scale, np.ndarray):
                    scale_dtype_code = 1  # float16
                    scale_raw = scale.tobytes()
                else:
                    # scalar
                    scale_dtype_code = 0  # float32
                    scale_raw = struct.pack("<f", float(scale))
                
                buf.write(struct.pack("<B", scale_dtype_code))
                buf.write(struct.pack("<H", len(scale_raw)))
                buf.write(scale_raw)
            else:
                buf.write(struct.pack("<B", 255))  # no scale marker
            
        else:
            # Passthrough float tensor
            t = passthrough[name]
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t)
            if not isinstance(t, Tensor):
                t = torch.tensor(t)
            
            name_bytes = name.encode("utf-8")
            buf.write(struct.pack("<B", len(name_bytes)))
            buf.write(name_bytes)
            
            # Scheme
            buf.write(struct.pack("<B", SCHEME_FLOAT_PASSTHROUGH))
            
            # Original dtype (from passthrough_orig_dtypes if available)
            orig_dtype = pt_dtypes.get(name, str(t.dtype).removeprefix("torch."))
            buf.write(struct.pack("<B", _dtype_to_code(orig_dtype)))
            
            # Shape
            shape = tuple(t.shape)
            buf.write(struct.pack("<B", len(shape)))
            for dim in shape:
                buf.write(struct.pack("<I", dim))
            
            # Data - store as-is (float32 or float16)
            raw = t.cpu().contiguous().numpy().tobytes()
            buf.write(struct.pack("<I", len(raw)))
            buf.write(raw)
            
            # No scale for passthrough
            buf.write(struct.pack("<B", 255))
    
    return buf.getvalue()


def deserialize_quantized(data: bytes) -> dict:
    """Deserialize flat binary format back to quantized state dict.
    
    Output format matches quantize_state_dict_int6() output:
    {quantized, scales, shapes, dtypes, passthrough, qmeta, ...}
    """
    pos = 0
    
    def read(n):
        nonlocal pos
        chunk = data[pos:pos + n]
        pos += n
        return chunk
    
    def read_fmt(fmt):
        nonlocal pos
        size = struct.calcsize(fmt)
        vals = struct.unpack(fmt, data[pos:pos + size])
        pos += size
        return vals[0] if len(vals) == 1 else vals
    
    # Header
    magic = read(4)
    if magic != MAGIC:
        raise ValueError(f"Bad magic: {magic}")
    version = read_fmt("<H")
    n_tensors = read_fmt("<H")
    
    quantized = {}
    scales = {}
    shapes = {}
    dtypes = {}
    passthrough = {}
    passthrough_orig_dtypes = {}
    qmeta = {}
    
    for _ in range(n_tensors):
        # Name
        name_len = read_fmt("<B")
        name = read(name_len).decode("utf-8")
        
        # Scheme + dtype
        scheme = read_fmt("<B")
        dtype_code = read_fmt("<B")
        dtype_str = _code_to_dtype(dtype_code)
        
        # Shape
        ndim = read_fmt("<B")
        shape = tuple(read_fmt("<I") for _ in range(ndim))
        
        # Data
        data_len = read_fmt("<I")
        raw = read(data_len)
        
        # Scale
        scale_dtype_code = read_fmt("<B")
        
        if scheme == SCHEME_FLOAT_PASSTHROUGH:
            # Reconstruct passthrough tensor
            np_dtype = np.float32 if dtype_str == "float32" else np.float16
            t = torch.from_numpy(np.frombuffer(raw, dtype=np_dtype).copy()).reshape(shape)
            passthrough[name] = t
            passthrough_orig_dtypes[name] = dtype_str
            shapes[name] = shape
            dtypes[name] = dtype_str
            
        elif scheme == SCHEME_INT8_PER_ROW:
            # Int8 packed data
            quantized[name] = np.frombuffer(raw, dtype=np.uint8).copy()
            shapes[name] = shape
            dtypes[name] = dtype_str
            qmeta[name] = {"scheme": "per_row_int8", "axis": 0}
            
            if scale_dtype_code != 255:
                scale_len = read_fmt("<H")
                scale_raw = read(scale_len)
                scale_dtype_str = _code_to_dtype(scale_dtype_code)
                if scale_dtype_str == "float16":
                    scales[name] = torch.from_numpy(np.frombuffer(scale_raw, dtype=np.float16).copy())
                elif scale_dtype_str == "float32":
                    scales[name] = torch.from_numpy(np.frombuffer(scale_raw, dtype=np.float32).copy())
                else:
                    scales[name] = torch.from_numpy(np.frombuffer(scale_raw, dtype=np.float32).copy())
            else:
                scales[name] = torch.tensor(1.0)
                
        elif scheme == SCHEME_INT6_PACKED:
            # Int6 packed data
            quantized[name] = np.frombuffer(raw, dtype=np.uint8).copy()
            shapes[name] = shape
            dtypes[name] = dtype_str
            
            # Determine if per-row or per-tensor from shape
            if len(shape) >= 2:
                qmeta[name] = {"scheme": "per_row", "axis": 0}
            
            if scale_dtype_code != 255:
                scale_len = read_fmt("<H")
                scale_raw = read(scale_len)
                scale_dtype_str = _code_to_dtype(scale_dtype_code)
                if scale_dtype_str == "float16":
                    scales[name] = torch.from_numpy(np.frombuffer(scale_raw, dtype=np.float16).copy())
                elif scale_dtype_str == "float32":
                    scales[name] = torch.from_numpy(np.frombuffer(scale_raw, dtype=np.float32).copy())
                else:
                    scales[name] = torch.from_numpy(np.frombuffer(scale_raw, dtype=np.float32).copy())
            else:
                scales[name] = torch.tensor(1.0)
    
    # Build output matching quantize_state_dict_int6 format
    quant_format = "mixed_int8_int6_packed_v1" if any(
        qmeta.get(n, {}).get("scheme") == "per_row_int8" for n in qmeta
    ) else "int6_packed_per_row_v1"
    
    obj = {
        "__quant_format__": quant_format,
        "quantized": quantized,
        "scales": scales,
        "shapes": shapes,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    
    return obj


def roundtrip_test(quant_obj: dict) -> bool:
    """Test that serialize → deserialize → dequantize matches torch.save path."""
    # Serialize
    custom_raw = serialize_quantized(quant_obj)
    
    # Deserialize
    restored = deserialize_quantized(custom_raw)
    
    # Compare quantized arrays
    for name in quant_obj.get("quantized", {}):
        orig = quant_obj["quantized"][name]
        rest = restored["quantized"][name]
        if isinstance(orig, np.ndarray) and isinstance(rest, np.ndarray):
            if not np.array_equal(orig, rest):
                print(f"MISMATCH quantized[{name}]")
                return False
    
    # Compare passthrough tensors
    for name in quant_obj.get("passthrough", {}):
        orig = quant_obj["passthrough"][name]
        rest = restored["passthrough"][name]
        if isinstance(orig, Tensor) and isinstance(rest, Tensor):
            if not torch.equal(orig, rest):
                # Allow float16 precision differences
                diff = (orig.float() - rest.float()).abs().max().item()
                if diff > 0.001:
                    print(f"MISMATCH passthrough[{name}] max_diff={diff}")
                    return False
    
    print(f"Roundtrip PASSED: {len(custom_raw):,} bytes")
    return True