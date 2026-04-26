#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data}"
SEED="${SEED:-1337}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-frontier_sources/scylla_pr1813/train_gpt.py}"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_pg}"
export TMPDIR="${TMPDIR:-/tmp}"
export RUN_ID="${RUN_ID:-scylla_qk525_loop3to5_bg40_seed${SEED}}"
export SEED
export ITERATIONS="${ITERATIONS:-9000}"
export VAL_LOSS_EVERY=0
export DATA_PATH="${DATA_PATH:-${DATA_ROOT}/datasets/fineweb10B_scylla}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${DATA_ROOT}/tokenizer/candidate.vocab}"
export TOKENIZER_META_PATH="${TOKENIZER_META_PATH:-${DATA_ROOT}/tokenizer/candidate.meta.npz}"
export VOCAB_SIZE=998
export XSA_LAST_N=11
export USE_GPTQ=1
export GPTQ_RESERVE_MS=9000
export TTT_ENABLED=0
export BIGRAM_VOCAB_SIZE=2816
export BIGRAM_DIM=40
export QK_GAIN_INIT=5.25
export NUM_LOOPS=2
export LOOP_START=3
export LOOP_END=5
export ENABLE_LOOPING_AT=0.35

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  env | grep -E '^(SEED|RUN_ID|ITERATIONS|VAL_LOSS_EVERY|DATA_PATH|TOKENIZER_PATH|TOKENIZER_META_PATH|VOCAB_SIZE|XSA_LAST_N|USE_GPTQ|GPTQ_RESERVE_MS|TTT_ENABLED|BIGRAM_VOCAB_SIZE|BIGRAM_DIM|QK_GAIN_INIT|NUM_LOOPS|LOOP_START|LOOP_END|ENABLE_LOOPING_AT)=' | sort
  echo "torchrun --standalone --nproc_per_node=8 ${TRAIN_SCRIPT}"
  exit 0
fi

cd "${ROOT_DIR}"
python3 "${ROOT_DIR}/scripts/check_scylla_assets.py" --root "${DATA_ROOT}"
python3 "${ROOT_DIR}/scripts/preflight_scylla.py"
torchrun --standalone --nproc_per_node=8 "${TRAIN_SCRIPT}"
python3 "${ROOT_DIR}/scripts/check_scylla_artifact.py" --model final_model.int6.ptz --code "${TRAIN_SCRIPT}" --min-margin 100000
