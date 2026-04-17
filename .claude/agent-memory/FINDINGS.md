# Parameter Golf — Verified Findings Log

This file is the team's ground truth. Only `[MEASURED]` results from actual runs belong here. Every agent should consult this before proposing new experiments.

---

## 2026-04-17 — 512d/13L MLP×4 WD=0.15 Full Run (Modal, 1×H100 kleanhard)

**Config:** dim=512, layers=13, mlp_mult=4, heads=8Q/4KV, depth_recurrence=[3,4,5]×2, parallel_residuals=True (start=7), XSA=all 13 layers, LeakyReLU²(slope=0.5), EMA(0.9965), OrthoInit, RoPE(16), LN_scale, smear, skip gates (scalar), parallel_final_lane_mean, SP8192 tokenizer, WD=0.15 (muon+embed+adam), GPTQ 64-batch calibration, embed_bits=8, clip_sigmas=12.85/20.0, custom serializer, NO byte shuffle, stride=2, brotli quality=11, TTT LoRA rank=96 on Q+K+O+MLP, 3-phase SGD TTT.

**Training:** 4500 steps, ~1.28s/step on 1×H100, seed=42.

**Results:**
- val_bpb trajectory: 1.3465 (500) → 1.2872 (1000) → 1.2627 (1500) → 1.2219 (2000) → 1.1763 (2500) → **1.1684 (3000, BEST)** → 1.1700 (3500) → 1.1861 (4000) → 1.2129 (4500)
- **Warmdown regression detected**: val_bpb degraded from 1.1684 to 1.2129 during warmdown (steps 3240-4500)
- Root cause: WD=0.15 too aggressive during LR warmdown — weights shrink when LR can't compensate

**Compression:**
- int6+brotli: 18,828,643 bytes (18.96 MB) — OVER BUDGET by 2.96 MB
- int6+zstd: 19,144,489 bytes (comparison)
- int8+zlib: 24,569,466 bytes (comparison)
- Compression ratio: 0.45 bytes/param (improved from 0.51 with WD=0.095, but still not enough)
- Payload ratio: 4.87× (32.5MB → 18.8MB)

**Verdict:** Best BPB (1.1684) beats our previous 640d best (1.1727), but artifact is 3MB over budget. Warmdown regression is a critical bug.

---

## 2026-04-17 — 384d/13L MLP×3 WD=0.15 Full Run (Modal, 1×H100 williguse)

**Config:** dim=384, layers=13, mlp_mult=3, heads=6Q/3KV, WD=0.15, same innovations as above.

**Training:** 2000 steps, ~0.935s/step on 1×H100.

**Results:**
- val_bpb: 1.3771 (500) → 1.2985 (1000) → 1.2465 (1500) → 1.2414 (2000)

**Compression:**
- int6+brotli: 10,470,000 bytes (10.47 MB) — FITS with 5.53 MB margin!
- Compression ratio: 0.40 bytes/param
- Payload ratio: 4.64× (16.3MB → 10.47MB)

**Verdict:** Guaranteed under 16MB but BPB too high (1.2414). Need more steps (4500+) and/or larger model.

---

## 2026-04-16 — 512d/13L MLP×4 WD=0.095 (Modal, 1×H100 williguse)

**Config:** Same as 512d WD=0.15 but with muon_wd=0.095, embed_wd=0.085, adam_wd=0.04.

**Training:** 2000 steps, ~1.29s/step.

**Results:**
- val_bpb: 1.3296 (500) → 1.2498 (1000) → 1.1887 (2000)

**Compression:**
- int6+brotli: 21,200,000 bytes (21.2 MB) — OVER BUDGET by 5.2 MB
- Compression ratio: 0.51 bytes/param

**Verdict:** WD=0.095 gives slightly better BPB at step 2000 (1.1887 vs 1.2219) but compresses much worse (21.2MB vs 18.96MB).

---

## 2026-04-14 — 768d/11L Full Training Run (Modal, 4×H100)

**Config:** dim=768, layers=11, mlp_mult=4, heads=12, depth_recurrence=[3,4,5]×2, parallel_residuals=True (start=7), XSA=[7,8,9,10], LeakyReLU²(slope=0.5), EMA(0.9965), SWA(0.995), OrthoInit, RoPE(16), LN_scale, smear, SP8192 tokenizer.

**Training:** 4500 steps, ~488ms/step, seed=42.

**Results:**
- val_bpb: 1.0882 (SWA, step 4500)

**Compression:**
- int6+brotli: 44,191,856 bytes (42.2 MB) — OVER BUDGET by 26.2 MB
- Payload ratio: 4.96×

**Verdict:** Incredible quality but completely over budget.

---

## 2026-04-14 — 640d/11L MLP×4 (Modal, 1×H100)

**Config:** dim=640, layers=11, mlp_mult=4, heads=10Q/5KV, WD=0.04, GPTQ-lite.

**Results:** val_bpb=1.1727 (step 2000, sliding eval exact)

**Compression:** 25.92 MB int6+brotli — OVER BUDGET by 9.92 MB

---

## 2026-04-14 — Compression Pipeline Experiments

**Byte shuffle diagnosis:**
- `[MEASURED]` Plain brotli without byte shuffle: 5.26 MB (untrained 640d)
- `[MEASURED]` Brotli with stride=2 shuffle: 7.00 MB (33% worse)
- `[MEASURED]` Brotli with stride=4 shuffle: 9.28 MB (76% worse)
- **Verdict:** Byte shuffle HURTS int6 compression. Removed from pipeline.

**Untrained vs trained compression gap:**
- `[MEASURED]` Untrained 768d int6+brotli: ~6.1 MB
- `[MEASURED]` Trained 768d int6+brotli: 42.2 MB
- **Ratio:** Trained weights compress ~7× worse than untrained

**Row-delta encoding:**
- `[MEASURED]` Row-delta + stride=3 on trained 512d WD=0.095: 22.7 MB (WORSE than no-delta)
- `[MEASURED]` Row-delta on untrained small model: neutral
- **Verdict:** Row-delta HURTS compression on trained weights. Disabled.

---

## Open Questions

1. ~~What is the trained compression ratio for a model in the 25-45M param range with the new pipeline?~~ → 0.45 bytes/param at 41.7M params, 0.40 at 20.4M params
2. What is the actual post-quant bpb degradation at clip_sigmas=12.85 with full GPTQ on our model?
3. ~~What architecture (dim/layers/mlp_mult) maximizes bpb quality while fitting in 16MB?~~ → 512d MLP×3 (~30M) or 640d MLP×3 (~46M)
4. Is mixed int5/int6 (int5 MLP, int6 attention) viable without unacceptable bpb loss?
5. What is the throughput on 8×H100 for a model sized to fit 16MB?
6. Does the WD warmdown schedule (0.15→0.095) actually prevent the regression?
7. What compression ratio can we achieve with WD=0.15 + WD warmdown schedule on 512d MLP×3?