# 🐉 OmniClaw — Multi-Agent System Implementation Master Plan

## Overview

Kai sent us a complete Claude Code multi-agent system with 6 specialist agents. Here's the deep analysis and plan to integrate it into our workflow.

---

## What We Received

### 1. CLAUDE.md (Conductor Brain)
- Mission statement, core disciplines, routing rules
- Agent roster with handle references
- Communication protocol (front-load context, structured output, Critic gate)
- Competition constraints (16MB, 10min, 8×H100)
- Updated with our latest measured results

### 2. Six Specialist Agents (.claude/agents/)
| Agent | File | Role | Key Trait |
|-------|------|------|-----------|
| Researcher | researcher.md | SOTA tracking, PR analysis | Primary sources only, never fabricate PR numbers |
| Architect | architect.md | Model design, sizing | Always calculates params+budget before proposing |
| Compressor | compressor.md | Quantization, serialization | Paranoid about trained vs untrained ratios |
| Trainer | trainer.md | Training loop, throughput | Thinks in ms/step, co-optimizes with quantization |
| Evaluator | evaluator.md | Validation, statistics | Distinguishes pre-quant vs post-quant, requires 3 seeds |
| Critic | critic.md | Adversarial review | 8 failure patterns, MANDATORY gate before presenting results |

### 3. Skill (SKILL.md)
- Competition rules, verified techniques, known failure modes
- Our measured results, decision framework

### 4. Findings Log (FINDINGS.md)
- Updated with all our measured results from April 14-17

---

## Critical Design Principles from the System

### Epistemic Labels (MUST follow)
- `[MEASURED]` — From an actual run with numbers
- `[ESTIMATED]` — Extrapolated from data
- `[PROPOSED]` — Untested idea
- `[SPECULATIVE]` — Theoretical only

### The 8 Failure Patterns (Critic watches for these)
1. **Untrained Weight Fallacy** — Testing compression on random weights
2. **Single-Tensor Extrapolation** — Micro-benchmark that doesn't compose
3. **Missing Quality Metric** — Size without bpb
4. **Victory Lap Before Validation** — "BREAKTHROUGH" before end-to-end test
5. **Serial Pivot** — Never closing the loop on a test
6. **Self-Arguing Message** — Wandering to different conclusions
7. **Fabricated Number** — Confabulated PR numbers or bpb scores
8. **Destructive Compression** — Great size, destroyed model

### Mandatory Critic Gate
EVERY result passes through Critic before presenting to Kai. No exceptions.

---

## Implementation Plan

### Phase 0: Integration (NOW)
- [x] Copy .claude/ directory to parameter-golf/
- [x] Update CLAUDE.md with our latest results
- [x] Update FINDINGS.md with all measured data
- [ ] Test that Claude Code recognizes the agents
- [ ] Update SKILL.md with our new measured compression ratios

### Phase 1: Run 512d MLP×3 with WD Warmdown (TODAY)
**Priority: HIGHEST — this is our next measurable data point**

**Config:**
```
DIM=512, LAYERS=13, MLP_MULT=3, NUM_HEADS=8, NUM_KV_HEADS=4
MUON_WEIGHT_DECAY=0.15 with WARMDOWN_WD_FRAC=0.37
WARMDOWN_FRAC=0.50 (shorter, less regression risk)
EMBED_BITS=8, GPTQ 64-batch, clip_sigmas=12.85/20.0
ITERATIONS=4500
```

**Expected:**
- ~30M params → ~9.2MB at 0.40 ratio (6.8MB margin)
- val_bpb target: ~1.10-1.12 (better than 384d's 1.2414)
- WD warmdown should prevent the regression we saw in v3

**Cost:** ~1×H100 × 97min × $3.92/hr ≈ $6.35

### Phase 2: Code Minification (TODAY, EASY)
- Compress train_gpt_kl.py + custom_serializer.py with LZMA
- SOTA uses this to shrink code from ~150KB to ~30KB
- **Frees 0.5-1MB for more model params**
- Implementation: write a compress_code.py that LZMA-compresses the source, add decompression to eval

### Phase 3: Per-Layer Adaptive Clip (THIS WEEK, MEDIUM)
- Different clip_sigmas per layer based on measured weight distributions
- SOTA #1626 does this
- Implementation: After training, measure per-layer weight stats, then re-quantize with per-layer clip
- **Expected: -0.3 to -0.5MB**

### Phase 4: Int7 Embeddings (THIS WEEK, MEDIUM)
- 7-bit packing: 8 values in 7 bytes (0.875 bytes/param)
- Between int6 and int8 for embeddings
- Implementation: Add pack_int7_rows_np/unpack_int7_rows_np to train_gpt_kl.py
- **Expected: -0.15MB over int8, +0.001 BPB quality over int6**

### Phase 5: Quantized Scales (THIS WEEK, MEDIUM)
- Store per-row scales as uint8 with shared max, instead of fp16
- Implementation: After quantization, convert scales from fp16 to uint8 with a global scale_max
- **Expected: -0.3MB**

### Phase 6: WD Warmdown Validation (AFTER Phase 1)
- Confirm the WD warmdown schedule actually prevents regression
- Compare v4 (512d MLP×3 + WD warmdown) vs v3 (512d MLP×4 + no WD warmdown)
- **This is the Critic gate: we must MEASURE that it works**

### Phase 7: 8×H100 Training (NEED CREDITS)
- Once we have GPU credits, run the optimal config on 8×H100
- 20K+ steps in 600s = proper convergence
- Should push ratio from 0.40→0.34 (SOTA-level compression)
- At 0.34 ratio, 640d MLP×3 (46M params) = 11.8MB ✅
- With TTT LoRA96, target: 1.07-1.09 BPB

---

## Agent Routing for Our Current Tasks

| Task | Agent | Why |
|------|--------|------|
| "Will 512d MLP×3 fit in 16MB?" | Architect | Param budget calculation |
| "Optimize compression pipeline" | Compressor | Quantization + serialization |
| "Fix warmdown regression" | Trainer | Training loop + optimizer |
| "Verify v4 results are real" | Critic | Adversarial review |
| "What's SOTA doing differently?" | Researcher | PR analysis |
| "Is this bpb improvement significant?" | Evaluator | Statistical analysis |

---

## The Conductor Protocol (How I Will Use This)

When Kai asks me a question, I will:

1. **Classify the task** → route to appropriate agent
2. **Front-load context** → give the agent everything it needs
3. **Request structured output** → `{size_bytes, val_bpb, config, epistemic_status}`
4. **Pass through Critic** → before presenting ANY result to Kai
5. **Update FINDINGS.md** → after every measured experiment

### Example: Kai asks "Will 512d MLP×3 fit?"

I invoke Architect with:
```
Context: Our latest FINDINGS.md data, current compression ratios, budget equation
Task: Calculate exact param count, estimated size at 0.40 and 0.38 ratios, budget headroom
Output: {total_params, est_size_040, est_size_038, margin_bytes, epistemic_status}
```

Then pass result through Critic for review before presenting.

---

## Self-Critique: Where We've Been Failing

Looking at our work through the Critic's 8 failure patterns:

1. ❌ **Untrained Weight Fallacy** — We tested compression on untrained weights multiple times
2. ⚠️ **Single-Tensor Extrapolation** — We estimated full-model compression from small models
3. ❌ **Missing Quality Metric** — We reported compression sizes without post-quant bpb
4. ❌ **Victory Lap Before Validation** — We celebrated "FITS UNDER 16MB!" before checking bpb
5. ⚠️ **Serial Pivot** — We jumped from idea to idea without closing loops
6. ⚠️ **Self-Arguing Message** — Long messages exploring options without ONE clear conclusion
7. ❌ **Fabricated Number** — We cited SOTA numbers without always verifying sources
8. ❌ **Destructive Compression** — We proposed WD=0.15 without measuring warmdown bpb impact

**This system will prevent all of these.** Going forward, every result gets:
- Epistemic label
- Both size AND bpb
- Critic review
- FINDINGS.md update

---

## Next Immediate Action

**Fire up 512d MLP×3 + WD warmdown on williguse H100.**

This gives us:
- A model that FITS under 16MB (guaranteed at 9.2MB)
- Better BPB than 384d (more params = more capacity)
- WD warmdown schedule to prevent regression
- A MEASURED end-to-end data point

Cost: ~$6.35 on williguse (~$5 remaining).

🫡