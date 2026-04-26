# Scylla Seed 1337 Modal Proof

Date: 2026-04-27 CEST
Branch: `codex/scylla-sub105-exec`
Commit: `641952b`
Modal app: `ap-STcs0VLgMY8HhRNGFcoDLD`
Modal URL: https://modal.com/apps/willi-guse26/main/ap-STcs0VLgMY8HhRNGFcoDLD

## Preflight

- Dataset repo: `amarck/parameter-golf-scylla`
- Dataset path: `/data/datasets/fineweb10B_scylla`
- Train shards: `194`
- Validation tokens reported by train script: `62363648`
- Modal smoke byte accounting: `val_tokens=62365135`, `val_bytes=182814497`, `tokens_per_byte=0.341138892`
- Tokenizer metadata: exact HF `tokenizer/candidate.meta.npz`, `source_model_name=scylla_tm0054`
- Flash attention interface: present

## Run

- Seed: `1337`
- Hardware: Modal `H100:8`
- Config: PR1813 Scylla QK 5.25, depth recurrence layers 3-5, `BIGRAM_DIM=40`, GPTQ int6, no TTT
- Train stop: `step=4923`, `train_time=591155ms`
- Post-EMA diagnostic: `val_loss=1.9529`, `val_bpb=0.9611`

## Artifact And Score

- Code size: `105865` bytes
- Compressed model: `15744092` bytes
- Total submission size: `15849957` bytes
- Decimal cap margin: `150043` bytes
- Final int6 roundtrip: `val_loss=1.95825204`, `val_bpb=0.96377557`
- Final int6 sliding window: `val_loss=1.91753823`, `val_bpb=0.94372928`

## Notes

The Modal runner used `frontier_sources/scylla_pr1813/train_gpt_modal.py`, which is derived from the PR1813 script with a rank-0-only post-GPTQ compression change. Training, quantized artifact contents, and evaluation semantics remain aligned with PR1813; the change avoids redundant quantize/LZMA work on non-writer ranks that caused the exact script to stall under the Modal timeout.
