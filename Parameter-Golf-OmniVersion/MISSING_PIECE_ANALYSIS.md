# Missing Piece Analysis: OpenAI Parameter Golf

## Current SOTA Landscape

| Rank | PR | Author | BPB | Key Innovations |
|------|-----|--------|-----|-----------------|
| 1 | #1530 | samacqua | 1.0734 | VarLen Attention + Fused MLP + **doc-independent LoRA TTT** |
| 2 | #1540 | aryanbhosale | 1.0777 | VarLen Attention + LoRA TTT + Fused MLP |
| 3 | #1493 | bigbag | 1.0810 | 3-Layer Recurrence + QK-Gain 5.25 |
| DQ | #1539 | translatingthename | 1.0587 | **DISQUALIFIED** — multi-epoch TTT on val data |

All top submissions share: SP8192 tokenizer, GPTQ SDClip quantization, depth recurrence, parallel residuals, Muon optimizer, score-first TTT.

## What Everyone Is Already Doing

- **Depth recurrence** (2-3 layer loops) — table stakes
- **Parallel residuals** (GPT-J style from layer 7+) — table stakes
- **GPTQ + SDClip** (int6 attention/MLP, int8 embeddings) — table stakes
- **Score-first TTT** (SGD/AdamW on chunks, 3-5 epochs) — table stakes
- **Muon optimizer** (row-normalized, Newton-Schulz) — table stakes
- **QK-Gain** (5.0-5.25) — near table stakes
- **EMA** (0.9965) — near table stakes
- **LeakyReLU²** — standard activation
- **Skip gates / U-Net** — standard

## Recent Innovations (already claimed)

- **VarLen Attention** (PR #1530, #1540) — attention over variable-length token sequences (presumably per-document boundaries)
- **Fused MLP** (PR #1530, #1540) — kernel-fused MLP for throughput
- **Doc-independent LoRA TTT** (PR #1530) — LoRA adapters trained per-document at TTT time
- **Banking** (PR #1518, #1523, #1533) — storing activations across recurrence passes
- **Muon 0.97** (PR #1521, #1541) — slightly higher momentum

## The ONE Missing Technique: **Asymmetric Entropy-Weighted TTT Learning Rate**

### What It Is

During score-first TTT, instead of using a uniform learning rate across all parameters and all token positions, weight the TTT gradient updates by the **local entropy** of the model's own predictive distribution at each position.

Specifically:
- For each position in a TTT chunk, compute `H_t = -Σ p_t(a) log p_t(a)` (the entropy of the model's prediction before adaptation)
- Use `lr_effective = lr_base * f(H_t)` where `f` is a monotone increasing function (e.g., `f(H) = 1 + α * (H - H_mean)`)
- Positions where the model is already confident (low entropy) get smaller TTT updates; uncertain positions (high entropy) get larger updates

### Why Nobody Has Tried It

1. **All current TTT implementations use uniform SGD/AdamW** — every token position gets the same learning rate
2. **The disqualified PR #1539 showed** that multi-epoch TTT on val data gets 1.0587 — a massive gain — but was illegal because it adapted too aggressively on validation tokens. The insight from that disqualification is: TTT *wants* to adapt more on hard/uncertain positions but *shouldn't* adapt on easy/certain ones (which risks overfitting/leakage)
3. **Entropy-weighted TTT is the legal version of "adapt more where it matters"** — it focuses TTT capacity on positions where the model is genuinely uncertain, without violating any of the four conditions

### Why It Could Give 0.005-0.01 BPB

1. **TTT is currently sub-optimal in its allocation** — it spends as much parameter update budget on tokens the model already predicts well as on tokens it doesn't. This wastes TTT capacity.
2. **Low-entropy positions are near-optimal already** — adapting on them provides marginal improvement but introduces overfitting risk (the model memorizes specific token sequences)
3. **High-entropy positions are where TTT adds the most value** — these are typically at document boundaries, rare words, or domain shifts where per-document adaptation matters most
4. **The entropy weighting is fully causal** — `H_t` depends only on `p_t(·)` which is computed from the artifact and the strict prefix, satisfying all four conditions
5. **The gain is bounded but meaningful** — we'd expect the effective TTT capacity to increase by ~30-50% on the positions that matter, yielding roughly the same improvement as going from 2-epoch TTT to 3-epoch TTT (which we know gives ~0.003-0.005 BPB), plus additional gain from reduced overfitting on easy positions

### Implementation Sketch

```python
# During TTT forward pass (scoring phase):
with torch.no_grad():
    logits = model(chunk_tokens)
    log_probs = F.log_softmax(logits, dim=-1)
    # Per-position entropy
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)  # [seq_len]
    entropy_mean = entropy.mean()

# During TTT backward pass:
# Weight the loss by entropy
alpha = 0.5  # hyperparameter
weights = 1.0 + alpha * (entropy - entropy_mean).clamp(min=-1.0)
weights = weights.detach()  # no gradient through weighting

# Per-token cross-entropy, weighted
token_losses = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction='none')
weighted_loss = (token_losses * weights).sum() / weights.sum()

# Backprop with weighted loss
weighted_loss.backward()
optimizer.step()
```

### Why This Is Legal

- **Condition 1 (Strict causal dependence):** The entropy `H_t` is computed from `p_t(·)`, which depends only on the artifact and prefix. ✓
- **Condition 2 (Full normalized distribution):** We're just weighting the loss; the output distribution is still a standard softmax over the full vocab. ✓
- **Condition 3 (Score before update):** Entropy is computed during the scoring pass, before any TTT update. The weighting is applied to the TTT loss, not to the scored distribution. ✓
- **Condition 4 (Single pass):** No multi-pass rescoring. ✓

### Synergy with Other Innovations

- **Doc-independent LoRA TTT** (PR #1530): Entropy weighting complements LoRA perfectly — low-entropy positions need smaller LoRA updates, high-entropy positions benefit from targeted adaptation
- **VarLen Attention**: Entropy naturally spikes at document boundaries where VarLen resets context — these are exactly the positions where TTT should adapt most
- **Banking**: Entropy-weighted TTT can use banking more effectively by prioritizing stored activations from high-entropy positions

## Alternative Candidates (Less Promising)

| Technique | Expected Gain | Risk | Why Deprioritized |
|-----------|--------------|------|-------------------|
| **rANS arithmetic coding post-processing** | 0.01-0.02 | **ILLEGAL** — violates Condition 2 (must output full normalized distribution over official vocab) | Not a neural technique; outputs a different representation |
| **Byte-shuffle + rANS on weights** | 0.001-0.002 | Low | Already done; marginal compression gains exhausted |
| **Pre-quant TTT on training data** | 0.003-0.005 | Medium | PR #1485, #1487 already explored; diminishing returns |
| **Compiled/fused TTT kernels** | 0.002-0.003 | Low | PR #1539 showed compiled TTT helps but the gain was largely from multi-epoch (illegal) |
| **Gated linear attention + retention** | 0.003-0.005 | Medium | Unproven in this regime; architectural risk |
| **Mixed-precision TTT** (fp32 LoRA, int6 base) | 0.001-0.003 | Low | Small gain, already somewhat implicit in doc-independent LoRA |
| **Data augmentation (token masking, BERT-style)** | 0.002-0.004 | Medium | Underexplored but hard to justify under score-first constraint |

## Recommended Implementation Priority

1. **Entropy-weighted TTT** (primary target, 0.005-0.01 BPB)
2. Combine with **doc-independent LoRA TTT** from PR #1530
3. Keep **VarLen Attention + Fused MLP + Banking + Muon 0.97** stack
4. Tune `α` (entropy weighting strength) via sweep on seed 42

## Estimated Final Score

Current best (PR #1530): **1.0734 BPB**

With entropy-weighted TTT on top of the #1530 stack: **1.063-1.068 BPB**

This would represent a legitimate (non-disqualified) record improvement, and the technique is novel — no PR has applied entropy-aware loss weighting during TTT.