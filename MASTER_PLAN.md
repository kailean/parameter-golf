# Parameter Golf: Sub-1.05 BPB & Sub-16MB Master Plan

## Current State (Apr 15, 2026)

### Our Best Run: 1×H100 640d MLP×4, 2000 steps
- **val_bpb: 1.1737** (standard eval), ~1.1714 (sliding eval, 75% done)
- **Compression: 25.92MB int6+brotli** ❌ OVER 16MB by 9.92MB
- **Params: 54.8M** (640d, 11L, MLP×4, GQA 10/5)
- Custom serializer saved 14MB vs torch.save but not enough
- Still converging at step 2000 (needs more steps)

### SOTA Leaderboard
| Rank | PR | BPB | Artifact | Key Techniques |
|------|-----|------|----------|----------------|
| 1 | #1626 | 1.07193 | 15.93MB | VarLen + Fused MLP + Multi-Phase Global SGD TTT + int7 embed + per-layer GPTQ clip |
| 2 | #1610 | 1.0728 | ~15.99MB | VarLen + Fused MLP + Phasing TTT |
| 3 | #1530 | 1.07336 | 15.99MB | VarLen + Fused MLP + Doc-Independent LoRA TTT |
| 4 | #1560 | 1.07406 | ~15.99MB | VarLen + Fused MLP + Doc-TTT + Warmdown 0.75 |
| 5 | #1586 | 1.07493 | ~15.99MB | Per-Layer Adaptive GPTQ Clip + int7 Embeddings + MLR 0.026 |
| 6 | #1493 | 1.0810 | ~15.5MB | 3-Layer Recurrence + QK-Gain 5.25 |

### Gap Analysis
- We need: **1.05 BPB** (target) → need **-0.12 BPB** improvement from current 1.1737
- SOTA is: **1.07193** → we need **-0.02 BPB** to beat it
- Our gap to SOTA: **0.10 BPB** (HUGE)
- Our compression gap: **25.92MB vs 15.99MB** (10MB over!)

---

## Root Cause: Why We're 10MB Over

### Compression Ratio Comparison
| Model | Params | int6+brotli | Bytes/Param | Notes |
|-------|--------|-------------|-------------|-------|
| SOTA #1530 | ~47M | 15.99MB | ~0.34 | GPTQ calibration, VarLen, fused MLP |
| SOTA #1626 | ~47M | 15.93MB | ~0.34 | Per-layer adaptive GPTQ, int7 embed |
| Ours 640d MLP×4 | 54.8M | 25.92MB | ~0.47 | GPTQ-lite only, no VarLen, no fused MLP |

**We have 17% more params but 62% more bytes.** Our bytes/param ratio is 38% worse than SOTA.

### Three Compression Killers:

1. **GPTQ-lite vs Full GPTQ** (BIGGEST factor)
   - Our GPTQ-lite only searches 5 percentile clip values per row
   - SOTA uses full GPTQ calibration (calibrates quantization against real data)
   - Full GPTQ dramatically reduces quantization error AND entropy
   - Lower entropy = better brotli compression
   - Estimated impact: **-5 to -8MB**

2. **Too many parameters (54.8M vs ~47M)**
   - SOTA fits ~47M params at 0.34 bytes/param = 15.99MB
   - We have 54.8M at 0.47 = 25.92MB
   - Even at SOTA's ratio: 54.8M × 0.34 = 18.6MB → STILL OVER
   - Need to reduce params to ~45M OR improve ratio to ~0.29
   - Estimated impact: **-3 to -4MB** (from reducing params)

3. **High entropy in trained weights**
   - Without proper GPTQ, trained int6 data has high entropy
   - High entropy = poor brotli compression
   - SOTA's per-layer adaptive clip further reduces entropy
   - Estimated impact: **-2 to -3MB** (from better entropy)

---

## The Path to Sub-1.05 BPB & Sub-16MB

### Phase 1: Fix Compression (Get Under 16MB) — Priority #1

Without fitting under 16MB, nothing else matters.

#### 1A. Full GPTQ Calibration (BIGGEST WIN)
- Replace GPTQ-lite with proper GPTQ calibration against validation data
- SOTA uses this; it's the single biggest compression win
- Reduces both quantization error (better BPB) AND entropy (better compression)
- **Expected: -5 to -8MB** → brings us from 25.92MB to ~18-21MB
- Still not enough alone, but massive step

#### 1B. Per-Layer Adaptive Clip (from PR #1586)
- Each layer gets its own clip percentile (not one global setting)
- Some layers need aggressive clipping, others don't
- Reduces entropy layer-by-layer
- **Expected: -1 to -2MB**

#### 1C. Reduce Model to ~45M Params
- Options:
  - 640d MLP×3 → 45.8M params (already tested: 12.51MB trained-like)
  - 640d MLP×4 with fewer layers (11→9) → ~42M
  - 512d MLP×4 → ~30M (too small?)
- At 0.34 bytes/param: 45.8M × 0.34 = 15.7MB ✅
- **Expected: -4 to -5MB** (from 54.8M → 45.8M)

#### 1D. int7 Embeddings (from PR #1626)
- Use 7-bit quantization for embeddings instead of 8-bit
- Better compression than int8, better quality than int6
- **Expected: -0.5 to -1MB**

#### 1E. Weight Decay as Compression Tool (from Issue #775)
- Higher weight decay → smaller weights → better compression
- WD=0.02: 15.73MB, WD=0.05: 13.56MB, WD=0.10: 10.98MB
- BUT: higher WD increases quantization gap (hurts BPB)
- The fix: **quantized scales** (uint8 + per-tensor scale_max) instead of float16
- This allows higher WD without the compression penalty from scale entropy
- **Expected: -2 to -4MB** (if quantized scales work)

**Phase 1 Total Expected: 25.92MB → ~12-14MB ✅**

### Phase 2: Improve BPB to Sub-1.05

#### 2A. VarLen Attention (from SOTA) — -0.005 to -0.01 BPB
- Pack documents with cu_seqlens boundaries
- Attention computed within documents only
- Reduces wasted FLOPs, cleaner training signal
- Requires Flash Attention 3 (FA3)
- **Impact: ~0.001 nats, ~2% faster training**

#### 2B. Fused MLP Triton Kernel (from SOTA) — -0.001 BPB
- Fuses up-projection + LeakyReLU(0.5)² + squaring
- Faster → more steps in 600s → better convergence
- **Impact: ~0.001 nats, ~3% faster training**

#### 2C. More Training Steps (8×H100)
- 1×H100 at 1.52s/step = 395 steps in 600s → val_bpb 1.2520
- 1×H100 at 2000 steps = 50 min → val_bpb 1.1737 (still converging!)
- 8×H100 ≈ 8× more steps in 600s ≈ 3160 steps → should reach ~1.12-1.15
- SOTA trains 587s on 8×H100 → much better convergence
- **Impact: -0.02 to -0.05 BPB**

#### 2D. Doc-Independent LoRA TTT (from SOTA) — -0.008 BPB
- Score-first LoRA adaptation on each document independently
- Strictly causal: score chunk i, then train on chunk i, then score chunk i+1
- 32-token chunks with batched LoRA
- **Impact: -0.008 nats**

#### 2E. Multi-Phase Global SGD TTT (from PR #1626) — -0.001 BPB
- Split prefix docs into 3 phases: score, SGD, score, SGD, score, SGD
- Progressively adapts while maintaining legality
- **Impact: -0.0008 BPB over single-phase TTT**

#### 2F. Warmdown 0.75 (from PR #1560)
- Extend warmdown to cover last 75% of training
- More gradual LR decay → better final weights
- **Impact: -0.001 to -0.003 BPB**

#### 2G. Better Optimizer Tuning
- MATRIX_LR=0.026 (from PR #1626)
- MUON_WD=0.095 → try 0.05-0.10 (compression + quality tradeoff)
- EMA decay tuning
- **Impact: -0.002 to -0.005 BPB**

### Phase 3: Push to Sub-1.05

#### 3A. PyMinifier / Code Compression — Free up ~500KB for model
- SOTA uses python-minifier to shrink code
- Each KB freed = more params for the model
- **Impact: +0.5MB model budget**

#### 3B. Scale Quantization (uint8 scales + scale_max)
- From Issue #775: store per-row scales as uint8 + per-tensor scale_max
- 1 byte per scale instead of 2 bytes (float16)
- Reduces scale storage by 50%
- **Impact: -0.5 to -1MB on scales**

#### 3C. 3-Seed Mean Submission
- SOTA uses 3 seeds and reports mean BPB
- Reduces variance, more stable number
- **Impact: more reliable, slight BPB improvement from variance**

---

## Execution Plan (Ordered by Impact × Feasibility)

### Week 1: Compression Fix (Days 1-3)
1. **Implement full GPTQ calibration** (replacing GPTQ-lite)
   - Use validation data for calibration
   - Per-layer adaptive clip
   - Test on existing 640d checkpoint
2. **Reduce to 640d MLP×3 (45.8M params)**
   - Re-train with compression-friendly config
   - Verify fits under 16MB
3. **Implement int7 embeddings**
4. **Test weight decay 0.05-0.10** for compression

### Week 2: BPB Improvement (Days 4-8)
5. **Get 8×H100 access** (RunPod credits, or use Modal williguse)
6. **Implement VarLen attention** (requires FA3)
7. **Implement fused MLP Triton kernel**
8. **Implement doc-independent LoRA TTT**
9. **Run full 600s competition training on 8×H100**

### Week 3: Polish (Days 9-14)
10. **Multi-phase SGD TTT**
11. **PyMinifier code compression**
12. **Scale quantization (uint8)**
13. **3-seed submission**

---

## Our Innovations (Beyond SOTA)

1. **Custom binary serializer** — 25% raw size reduction vs torch.save
2. **Quantized scales (uint8)** — from Issue #775, not yet implemented by anyone
3. **Higher weight decay + quantized scales** — novel compression strategy
4. **Entropy-aware quantization** — adapt quantization strategy per-layer based on entropy analysis

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| GPTQ calibration doesn't reduce enough | Medium | High | Fall back to 640d MLP×3 + higher WD |
| No 8×H100 access | High | High | Use 4×H100, apply for more credits |
| VarLen + FA3 incompatible with our code | Medium | Medium | Use standard attention, lose ~0.005 BPB |
| TTT doesn't improve our score | Low | Medium | Our earlier TTT hurt (1.73 vs 1.18), need doc-independent |
| Deadline pressure (Apr 30) | Medium | High | Focus on compression first, then BPB |