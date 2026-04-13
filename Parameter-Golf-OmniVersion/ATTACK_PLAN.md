# 🐉⚡ OmniClaw Parameter Golf — #1 Attack Plan
## Target: Beat 1.0734 BPB (current SOTA, PR #1530)
## Secret Weapon: Entropy-Weighted TTT

---

## Competition State (April 11, 2026 — LIVE)

| Rank | BPB | Key Stack | PR | Status |
|------|-----|-----------|-----|--------|
| 🥇 | 1.0734 | VarLen + Fused MLP + doc-independent LoRA TTT | #1530 | ✅ OPEN |
| 🥈 | 1.0777 | VarLen + LoRA TTT + Fused MLP | #1540 | ✅ OPEN |
| 🥉 | 1.0810 | 3-Layer Recurrence + QK-Gain 5.25 | #1493 | ✅ MERGED |
| ~~DQ~~ | ~~1.0587~~ | ~~Pre-Quant TTT~~ | ~~#1539~~ | ❌ DISQUALIFIED |
| **Ours** | **1.1233** | v3 on 8×H100 | **#414** | ❌ CLOSED |

**Target: 1.0734 BPB → beat with entropy-weighted TTT**
**Estimated final: 1.063-1.068 BPB**

---

## Our Novel Innovation: Entropy-Weighted TTT

During score-first TTT, weight gradient updates by local predictive entropy:
- High-entropy (uncertain) positions → larger TTT updates
- Low-entropy (confident) positions → smaller TTT updates
- **Fully legal**: entropy computed from prefix-only predictions (causal)
- **Expected gain: 0.005-0.01 BPB** on top of the #1530 stack
- **No PR has tried this** — all TTT implementations use uniform learning rate

### Implementation
```python
# Score pass: compute entropy
with torch.no_grad():
    logits = model(chunk_tokens)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    entropy_mean = entropy.mean()

# TTT pass: weight loss by entropy
alpha = 0.5
weights = 1.0 + alpha * (entropy - entropy_mean).clamp(min=-1.0)
token_losses = F.cross_entropy(logits.view(-1, V), targets.view(-1), reduction='none')
weighted_loss = (token_losses * weights.detach()).sum() / weights.detach().sum()
```

---

## The Winning Stack (What #1 Uses)

1. SP8192 tokenizer (8192 vocab)
2. 11-layer, 512-dim, 8 attention heads, 4 KV heads
3. Tied embeddings (init_std=0.005)
4. 4× MLP, partial RoPE (16/64)
5. QK-Gain 5.25
6. Muon optimizer (momentum=0.99, Newton-Schulz)
7. EMA 0.9965, warmdown 72%
8. GPTQ SDClip (int6 matrices, int8 embeddings)
9. Depth recurrence (L3-5, 3×)
10. Parallel residuals (L7+)
11. Legal score-first TTT (doc-independent LoRA)
12. Banking (3D weight stacking)
13. Brotli compression

---

## Phase 1: Reproduce + Entropy TTT (Days 1-3)
- [ ] Set up RunPod 8×H100
- [ ] Reproduce #1530 baseline (~1.073 BPB)
- [ ] Implement entropy-weighted TTT
- [ ] Smoke test on M4

## Phase 2: Optimize + Sweep (Days 4-7)
- [ ] Sweep alpha (entropy weight strength): 0.0, 0.25, 0.5, 0.75, 1.0
- [ ] Sweep TTT LoRA rank: 64, 96, 128
- [ ] Sweep QK-Gain: 5.0, 5.25, 5.5
- [ ] Try VarLen attention + Fused MLP

## Phase 3: Final Submission (Days 8-14)
- [ ] 3-seed verification
- [ ] Artifact < 16MB check
- [ ] PR submission

## Hardware & Data
- RunPod 8×H100 ($1,000 credits pending)
- SP8192 data: 75/128 shards downloaded, rest downloading
- SP8192 tokenizer: ✅ verified working
- Internal SSD: 9GB free (HF cache on external SSD)

## Compute Credit Application
- Applied: Advanced Competitor ($1,000 / ~320 hours)
- Email: kai.leanhard@hotmail.com
- GitHub: kailean
- Status: Pending (1-2 business days)