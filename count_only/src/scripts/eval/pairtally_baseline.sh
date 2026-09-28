#!/bin/bash

# Evaluate CountEx baseline (without weakly-supervised training) on PairTally
# This gives inter/intra breakdown for the baseline

export TOKENIZERS_PARALLELISM="false"
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"

# CountEx baseline checkpoint (Stage 1, no weakly-supervised finetuning)
CKPT_PATH="BBVisual/CountEX-KC"
GPU_ID=7

cd "$(dirname "$0")/../.."

python eval_pairtally_stage2.py \
    --ckpt_path ${CKPT_PATH} \
    --model_type countex_stage2 \
    --gpu_id ${GPU_ID} \
    --save_root ${EXP_ROOT:-./experiments}
