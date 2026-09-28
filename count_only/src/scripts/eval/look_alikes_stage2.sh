#!/bin/bash

export TOKENIZERS_PARALLELISM="false"
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"

# Settings
WEEK="W8"
EXPERIMENT_NAME="kc_countex_stage2_100k_666"
CKPT_PATH="${EXP_ROOT:-./experiments}/${WEEK}/${EXPERIMENT_NAME}/best_val_model"
GPU_ID=7

cd "$(dirname "$0")/../.."

python eval_look_alikes_with_neg_stage2.py \
    --ckpt_path ${CKPT_PATH} \
    --model_type countex_stage2 \
    --gpu_id ${GPU_ID} \
    --save_root ${EXP_ROOT:-./experiments}
