#!/usr/bin/env bash
# =============================================================================
# Evaluate a trained NeuroStat checkpoint on a single pair-format JSON file.
#
# Example:
#   MODEL_PATH=./ckpt/neurostat \
#   EVAL_DATA=data/test.json \
#   SAVE_PATH=./results/neurostat \
#   SAVE_FILE=test.json \
#   bash scripts/eval_neurostat.sh
# =============================================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH=${MODEL_PATH:-${ROOT_DIR}/ckpt/neurostat}
EVAL_DATA=${EVAL_DATA:?"Set EVAL_DATA to a pair-format JSON file."}
EVAL_DATA_FORMAT=${EVAL_DATA_FORMAT:-MIRAGE}
SAVE_PATH=${SAVE_PATH:-${ROOT_DIR}/results/neurostat}
SAVE_FILE=${SAVE_FILE:-eval.json}
THRESHOLD=${THRESHOLD:-0.5}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-512}
USE_CPU=${USE_CPU:-0}
SKIP_BEST_THRESHOLD=${SKIP_BEST_THRESHOLD:-1}

if [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] Checkpoint directory does not exist: ${MODEL_PATH}"
    echo "Run scripts/train_neurostat.sh first, or set MODEL_PATH to a trained checkpoint."
    exit 1
fi

EXTRA_ARGS=""
if [ "${USE_CPU}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --use_cpu"
fi
if [ "${SKIP_BEST_THRESHOLD}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --skip_best_threshold"
fi

python ${ROOT_DIR}/eval.py \
    --model_name_or_path "${MODEL_PATH}" \
    --eval_data_path "${EVAL_DATA}" \
    --eval_data_format "${EVAL_DATA_FORMAT}" \
    --save_path "${SAVE_PATH}" \
    --save_file "${SAVE_FILE}" \
    --eval_batch_size "${BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --threshold "${THRESHOLD}" \
    ${EXTRA_ARGS}
