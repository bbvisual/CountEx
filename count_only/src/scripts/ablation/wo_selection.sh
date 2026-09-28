#!/bin/bash

# Stage 2: Finetune dot-pretrained CountEX with detection prior + Mean Teacher
# All parameters trainable, detection prior generated from decoder outputs
# Mean Teacher: student-teacher EMA for consistency regularization

export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++
export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TOKENIZERS_PARALLELISM="false"
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
export WANDB_API_KEY="${WANDB_API_KEY:-}"   # optional, leave empty to disable W&B
export HF_TOKEN="${HF_TOKEN:?export your HuggingFace token first}"
export WANDB_DIR="${WANDB_DIR:-./wandb}"
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"

SIZE="tiny"
PROJECT_NAME="ECCV_Experiments_kc"
EXPERIMENT_NAME="ablation_wo_selection_kc_countex_stage2_100k_666"
MODEL="countex_stage2"
DATA_SPLIT="ALL"
WEEK="W8"

# Dot-pretrained CountEX checkpoint (Stage 1)
PRETRAINED_PATH="BBVisual/CountEX-KC"

export WANDB_PROJECT="${PROJECT_NAME}"
export WANDB_NAME="${EXPERIMENT_NAME}"

accelerate launch --main_process_port 29502 --config_file ./ddp_cfgs/1n6r.yaml train_alternating.py \
    --backbone_size "${SIZE}" \
    --model "${MODEL}" \
    --pretrained_path "${PRETRAINED_PATH}" \
    --save_qualitative_results False \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --weight_decay 0.00001 \
    --logging_steps 50 \
    --eval_steps 500 \
    --save_steps 2000 \
    --dataloader_num_workers 4 \
    --output_dir "${EXP_ROOT:-./experiments}/${WEEK}/${EXPERIMENT_NAME}" \
    --train_data_path "BBVisual/CoCount-train-2" \
    --val_data_path "BBVisual/CoCount-val" \
    --test_data_path "BBVisual/CoCount-test" \
    --weakly_supervised_data_path "BBVisual/FG-Count-V2-weak-sup-train" \
    --save_total_limit 3 \
    --remove_unused_columns False \
    --dataloader_pin_memory False \
    --bf16 True \
    --report_to "wandb" \
    --run_name "${EXPERIMENT_NAME}" \
    --wandb_name "${EXPERIMENT_NAME}" \
    --wandb_project "${PROJECT_NAME}" \
    --lr_scheduler_type "constant" \
    --seed 666 \
    --data_split "${DATA_SPLIT}" \
    --weakly_supervised_sample_num 1 \
    --use_weakly_supervised_training True \
    --use_uncertainty_loss False \
    --count_loss_type "interval_huber" \
    --use_neg_prob 0.0 \
    --use_pseudo_density_loss True \
    --pseudo_density_weight 500.0 \
    --use_alternating_training True \
    --density_phase_length 100 \
    --detection_phase_length 10 \
    --detection_alignment_weight 0.005 \
    --detection_count_loss_weight 0.005 \
    --use_mean_teacher True \
    --consistency_weight 500.0 \
    --ema_decay 0.999 \
    --max_grad_norm 1.0 \
    --train_data_ratio 0.5
