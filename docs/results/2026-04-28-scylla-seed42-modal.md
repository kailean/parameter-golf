# Scylla Seed 42 Modal Proof

Date: 2026-04-28 CEST
Branch: `codex/scylla-sub105-exec`
Modal app: `ap-2UnjXHq87JCsCdAIwF5JRw`

## Run

- Seed: `42`
- Hardware: Modal `H100:8`
- Config: PR1813 Scylla QK 5.25, depth recurrence layers 3-5, `BIGRAM_DIM=40`, GPTQ int6, no TTT
- Train stop: `step=4693`, `train_time=591149ms`
- Post-EMA diagnostic: `val_loss=1.9553`, `val_bpb=0.9623`

## Artifact And Score

- Code size: `105865` bytes
- Compressed model: `15745296` bytes
- Total submission size: `15851161` bytes
- Decimal cap margin: `148839` bytes
- Final int6 roundtrip: `val_loss=1.96081623`, `val_bpb=0.96503757`
- Final int6 sliding window: `val_loss=1.92003455`, `val_bpb=0.94495785`

## Notes

This run is valid and comfortably sub-1.05, but slower than the PR1813 seed-42 reference because the Modal worker reached only 4,693 training steps before the wallclock stop. PR1813 seed 42 reached 5,260 steps. The result is useful for robustness, but a faster-capacity rerun may be needed for a best 3-seed record package.
