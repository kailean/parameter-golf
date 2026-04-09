# Competition Status & Next Steps
## Updated: 2026-04-09 23:50 CEST

### ✅ Compute Grant Application Submitted
- Applied for Advanced Competitor Grant ($1000 / ~320 H100 hours)
- Email: kai.leanhard@hotmail.com
- GitHub: kailean
- Country: Switzerland
- Status: Pending review (1-2 business days)

### Current Results
| Run | val_bpb | Notes |
|-----|---------|-------|
| v3 Baseline | 2.7612 | 11L, 512d, Muon, EMA, int6 QAT |
| Moonshot Phase 1 | 2.8168 | +EngramLite +SkipGram +Complementary +NgramMixer (WORSE) |

### Competition Landscape
| Rank | BPB | Technique |
|------|-----|-----------|
| 1 | 0.1156 | Dirichlet 15-gram + phrase cache + EBLS |
| 2 | 0.978 | Leaky TTT (invalid) |
| 3 | 1.0226 | Gated DeltaNet (buggy eval) |
| 4 | 1.1086 | signalrush: SOTA clean entry |
| Us | 2.7612 | M4 baseline, no GPU |

### Priority Actions
1. **Wait for Runpod credits** (1-2 days)
2. **Implement Dirichlet n-gram mixer** (orders 2-15, per-order concentrations)
3. **Implement phrase cache** (exact suffix matching)
4. **Port to PyTorch/CUDA** for 8×H100
5. **First GPU run → measure real BPB**

### Key Insight
The competition is an n-gram compression challenge. Pure neural ≈ 1.10 BPB.
The remaining 0.85+ BPB comes from eval-time n-gram mixing.
Our code already has BackoffNgramMixer — we need to extend it to
orders 2-15 with Dirichlet posterior smoothing.