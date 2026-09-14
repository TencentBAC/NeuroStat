#!/usr/bin/env bash
# =============================================================================
# Train NeuroStat end-to-end. All hyper-parameters can be overridden by
# environment variables.
#
# Example:
#   MODEL_NAME_OR_PATH=Qwen/Qwen2-0.5B \
#   TRAIN_DATA=data/train.json \
#   OUTPUT_DIR=./ckpt/neurostat \
#   bash scripts/train_neurostat.sh
# =============================================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- Model / Data ----
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-Qwen/Qwen2-0.5B}
CACHE_DIR=${CACHE_DIR:-./cache}
TRAIN_DATA=${TRAIN_DATA:-${ROOT_DIR}/data/train.json}
TRAIN_DATA_FORMAT=${TRAIN_DATA_FORMAT:-MIRAGE}
EVAL_DATA=${EVAL_DATA:-}            # optional
EVAL_DATA_FORMAT=${EVAL_DATA_FORMAT:-MIRAGE}

# ---- Output ----
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT_DIR}/ckpt/neurostat}

# ---- Training flags ----
USE_CPU=${USE_CPU:-0}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-0}
DISABLE_EVAL_DURING_TRAINING=${DISABLE_EVAL_DURING_TRAINING:-1}
SKIP_FINAL_EVAL=${SKIP_FINAL_EVAL:-1}

# ---- Architecture dims ----
TF_DIM=${TF_DIM:-64}
TB_DIM=${TB_DIM:-64}

# ---- Loss weights ----
LAMBDA_SUPCON=${LAMBDA_SUPCON:-0.1}
SUPCON_TEMPERATURE=${SUPCON_TEMPERATURE:-0.07}
LAMBDA_ORTH=${LAMBDA_ORTH:-0.001}

# ---- Training hyper-params ----
NUM_EPOCHS=${NUM_EPOCHS:-5}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_LENGTH=${MAX_LENGTH:-512}

EXTRA_ARGS=""
if [ "${USE_CPU}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --use_cpu"
fi
if [ "${DISABLE_EVAL_DURING_TRAINING}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --disable_eval_during_training"
fi
if [ "${SKIP_FINAL_EVAL}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --skip_final_eval"
fi
if [ -n "${EVAL_DATA}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --eval_data_path ${EVAL_DATA} --eval_data_format ${EVAL_DATA_FORMAT}"
fi

python ${ROOT_DIR}/train.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --cache_dir "${CACHE_DIR}" \
    --train_data_path "${TRAIN_DATA}" \
    --train_data_format "${TRAIN_DATA_FORMAT}" \
    --output_dir "${OUTPUT_DIR}" \
    --freeze_backbone "${FREEZE_BACKBONE}" \
    --tf_dim "${TF_DIM}" \
    --tb_dim "${TB_DIM}" \
    --lambda_supcon "${LAMBDA_SUPCON}" \
    --supcon_temperature "${SUPCON_TEMPERATURE}" \
    --lambda_orth "${LAMBDA_ORTH}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --eval_batch_size 8 \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --max_length "${MAX_LENGTH}" \
    ${EXTRA_ARGS}
