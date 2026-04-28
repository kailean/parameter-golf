# SOTA Max Design

## Objective

Push beyond the clean Scylla `0.94166` record while keeping a defensible path for review. The target is two-tiered: first produce a clean 3-seed Scylla reproduction that beats or matches PR #1813, then run a higher-upside token-level PPM mixture lane that can compete with the current open PR claims around `0.85` to `0.82` without inheriting the byte-level PPM legality problem.

## Current Baseline

Our branch `codex/scylla-sub105-exec` has a Modal 8xH100 seed-1337 proof:

- `final_int6_roundtrip_exact val_bpb=0.96377557`
- `final_int6_sliding_window_exact val_bpb=0.94372928`
- total submission size `15,849,957` bytes
- exact Scylla tokenizer metadata `source_model_name=scylla_tm0054`

This is already sub-1.05 and close to PR #1813. The remaining gap to PR #1813 on seed 1337 is mostly throughput/noise: our run stopped at step `4923` versus PR #1813 seed-1337 at step `5249`.

## Strategy

### Lane A: Defensible Scylla++

Complete the clean three-seed Scylla proof, then make narrow safe changes that preserve PR #1813's legal surface:

1. Run seeds `42` and `2025` with the exact Scylla metadata and rank-0 Modal compression fix.
2. Compare per-seed step count, post-EMA BPB, roundtrip BPB, sliding BPB, artifact bytes, and train/eval time against PR #1813.
3. Add a controlled ablation launcher for safe knobs only: `BIGRAM_DIM`, `QK_GAIN_INIT`, loop enable fraction, and loop range.
4. Promote only changes that improve seed `1337` without reducing cap margin below `100,000` bytes or increasing compliance risk.

Expected upside: small but defensible, likely `0.001` to `0.005` BPB.

### Lane B: Maximum-Upside Token-Level PPM

Build a PPM mixture that addresses the open Issue #1872 objection directly. The questionable current PR cluster mixes byte-level PPM probabilities at byte positions; the maintainer objection is that the resulting probability mass does not sum to one over the official token alphabet. Our version should instead construct a normalized distribution over token IDs:

1. For each candidate token in the tokenizer vocabulary, decode the token into its byte contribution under the same byte-accounting LUT used by BPB.
2. Score that byte sequence under a causal PPM state that contains only already-scored prefix bytes.
3. Normalize candidate token probabilities over the token vocabulary.
4. Mix in token probability space: `p_mix(token) = lambda * p_nn(token) + (1 - lambda) * p_ppm_token(token)`.
5. Score the true token, then update PPM state with the true token bytes.

This keeps the mixture causal, single-pass, score-before-update, and normalized over token IDs. It is slower than byte-level PPM, so we need a staged implementation: CPU/local correctness tests first, Modal smoke on a small validation prefix second, then full seed only if timing is plausible.

Expected upside: large if legal and fast enough, plausibly `0.05` to `0.12` BPB on top of Scylla. This is the best route to beating the `0.85` open PR cluster with a more defensible argument.

## Guardrails

- Do not mutate the original `train_gpt_kl.py` lane.
- Keep all experiments in `codex/scylla-sub105-exec` or a child worktree.
- Do not commit generated model artifacts or fresh GPU logs unless they are curated result docs.
- Keep `16,000,000` byte decimal cap checks mandatory.
- Treat byte-level PPM, lossy casefolding, SLOT, ETLB, pre-quant validation TTT, and non-normalized scalar mixtures as high-risk unless explicitly isolated as non-record experiments.
- Use exact Scylla metadata and report `tokens_per_byte` in every proof.

## Testing

Local tests must cover metadata byte accounting, artifact size checks, runner environment rendering, and token-level PPM normalization. Modal smoke must verify assets, tokenizer metadata, byte factor, FlashAttention availability, and a small-prefix PPM correctness/timing check before full runs.

## First Execution Batch

1. Run seeds `42` and `2025` with the current Scylla runner.
2. Record a 3-seed Scylla result doc.
3. Add a sweep-capable Modal entrypoint for safe Scylla++ knobs.
4. Add local token-level PPM normalization tests and a small standalone scorer before touching the Scylla training script.
