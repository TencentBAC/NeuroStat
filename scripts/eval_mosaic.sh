#!/usr/bin/env bash
# =============================================================================
# Evaluate a trained NeuroStat checkpoint on all 8 MOSAIC adversarial
# categories and dump per-category JSONs.
#
# The MOSAIC benchmark is shipped with this repository under ./benchmark/:
#   ${MOSAIC_DIR}/
#     ├── cat1_interleaved_human_ai.json
#     ├── cat2_statistical_hijacking.json
#     ├── cat3_translation_laundering.json
#     ├── cat4_paraphrase_polish.json
#     ├── cat5_prompt_injection.json
#     ├── cat6_character_encoding.json
#     ├── cat7_recursive_multipass.json
#     └── cat8_domain_format.json
#
# Each file follows the "MIRAGE" pair format:
#   [{"original": "...", "rewritten": "..."}, ...]
# =============================================================================
set +e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH=${MODEL_PATH:-${ROOT_DIR}/ckpt/neurostat}
MOSAIC_DIR=${MOSAIC_DIR:-${ROOT_DIR}/benchmark}
SAVE_PATH=${SAVE_PATH:-${ROOT_DIR}/results/neurostat_mosaic}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_LENGTH=${MAX_LENGTH:-512}
THRESHOLD=${THRESHOLD:-0.5}

CAT_FILES=(
    "cat1_interleaved_human_ai"
    "cat2_statistical_hijacking"
    "cat3_translation_laundering"
    "cat4_paraphrase_polish"
    "cat5_prompt_injection"
    "cat6_character_encoding"
    "cat7_recursive_multipass"
    "cat8_domain_format"
)
CAT_SHORT=(cat1 cat2 cat3 cat4 cat5 cat6 cat7 cat8)

mkdir -p "${SAVE_PATH}"

for i in "${!CAT_FILES[@]}"; do
    CAT="${CAT_SHORT[$i]}"
    DATA="${MOSAIC_DIR}/${CAT_FILES[$i]}.json"
    if [ ! -f "${DATA}" ]; then
        echo "[SKIP] Missing MOSAIC file: ${DATA}"
        continue
    fi

    echo ">>> NeuroStat / ${CAT}"
    python ${ROOT_DIR}/eval.py \
        --model_name_or_path "${MODEL_PATH}" \
        --eval_data_path "${DATA}" \
        --eval_data_format MIRAGE \
        --save_path "${SAVE_PATH}" \
        --save_file "MOSAIC_${CAT}.json" \
        --eval_batch_size "${BATCH_SIZE}" \
        --max_length "${MAX_LENGTH}" \
        --threshold "${THRESHOLD}" \
        --skip_best_threshold
done

echo ""
echo "============================================================"
echo "MOSAIC benchmark evaluation done."
echo "Results saved to: ${SAVE_PATH}/"
echo "============================================================"
