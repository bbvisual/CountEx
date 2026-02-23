#!/bin/bash

export CC=$CONDA_PREFIX/bin/gcc
export CXX=$CONDA_PREFIX/bin/g++
export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TOKENIZERS_PARALLELISM="false"
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export WANDB_API_KEY=""
export HF_TOKEN=""

DEEPSPEED_CONFIG="ddp_cfgs/zero2.json"
SIZE="tiny"
PROJECT_NAME="CountEx_NC_Home"
EXPERIMENT_NAME="CountEx_NC_Home"
MODEL="countex"
DATA_SPLIT="HOU"

export WANDB_PROJECT="${PROJECT_NAME}"
export WANDB_NAME="${EXPERIMENT_NAME}"

accelerate launch --main_process_port 29500 --config_file ./ddp_cfgs/1n4r.yaml train.py \
    --backbone_size "${SIZE}" \
    --model "${MODEL}" \
    --save_qualitative_results False \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.00001 \
    --warmup_steps 0 \
    --logging_steps 50 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --dataloader_num_workers 4 \
    --output_dir "/data/add_disk0/yifengc/countex_exp" \
    --train_data_path "BBVisual/CoCount-train" \
    --val_data_path "BBVisual/CoCount-val" \
    --test_data_path "BBVisual/CoCount-test" \
    --weakly_supervised_data_path "BBVisual/CoCount-train" \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --dataloader_pin_memory False \
    --bf16 True \
    --report_to "wandb" \
    --run_name "${EXPERIMENT_NAME}" \
    --lr_scheduler_type "constant" \
    --use_weakly_supervised_training False \
    --seed 666 \
    --data_split "${DATA_SPLIT}" \
    --weakly_supervised_sample_num 1
