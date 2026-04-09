# Parameter Golf — Competition Submission Plan
## Team: KaiLean (@kailean) + OmniClaw
## Email: kai.leanhard@hotmail.com
## GitHub: kailean

### Strategy
1. Apply for OpenAI compute grant (Advanced competitor — $1000 / ~320 H100 hours)
2. Port train_gpt_kl.py (PyTorch/CUDA) to 8×H100
3. Implement Dirichlet posterior n-gram mixing (orders 2-15)
4. Implement phrase cache (exact suffix matching)
5. Run, measure, submit

### Compute Grant Application
- Email: kai.leanhard@hotmail.com
- GitHub: kailean
- Level: Advanced competitor ($1000 / ~320 hours)
- Justification: We have a working training pipeline with Muon optimizer,
  int6 QAT, EMA, XSA, BigramHash, SmearGate, and BackoffNgramMixer.
  Our current M4 baseline scores 2.76 BPB. With 8×H100 access we can
  reach competitive territory (~1.10 pure neural, ~0.30 with n-gram mixing).