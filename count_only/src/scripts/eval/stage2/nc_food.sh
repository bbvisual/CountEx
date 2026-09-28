#!/bin/bash

export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="0"
export WANDB_API_KEY="${WANDB_API_KEY:-}"   # optional, leave empty to disable W&B
# export HF_TOKEN="${HF_TOKEN:?export your HuggingFace token first}"
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"

# Settings
MODEL="countex_stage2"
SIZE="tiny"
DATA_SPLIT="FOO"
WEEK="W8"
EXPERIMENT_NAME="nc_food_countex_stage2_100k_666"
CKPT_PATH="${EXP_ROOT:-./experiments}/${WEEK}/${EXPERIMENT_NAME}/best_val_model"

if [ -z "$CKPT_PATH" ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 ${EXP_ROOT:-./experiments}/W8/nc_food_countex_stage2_100k_666/best_val_model"
    exit 1
fi

cd "$(dirname "$0")/../.."

# Run evaluation
python eval_density.py \
    --model ${MODEL} \
    --backbone_size ${SIZE} \
    --ckpt_path ${CKPT_PATH} \
    --train_data_path "BBVisual/CoCount-train" \
    --val_data_path "BBVisual/CoCount-val" \
    --test_data_path "BBVisual/CoCount-test" \
    --weakly_supervised_data_path "BBVisual/CoCount-train" \
    --output_dir "${CKPT_PATH}/eval_results" \
    --data_split ${DATA_SPLIT} \
    --batch_size 1 \
    --save_visualizations True \
    --max_vis_samples 100

echo "Evaluation completed! Results saved to ${CKPT_PATH}/eval_results"
