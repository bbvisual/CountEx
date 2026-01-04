#!/bin/bash

export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++
export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TOKENIZERS_PARALLELISM="false"
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,6,7"
export WANDB_DIR="/nfs/130.245.4.102/add_disk0/yifeng/wandb/"
export HF_HOME="/nfs/130.245.4.102/add_disk3/yifeng/hug_ckpt"

DEEPSPEED_CONFIG="ddp_cfgs/zero2.json"
SIZE="tiny"
PROJECT_NAME="CountEx_KC"
EXPERIMENT_NAME="kc_auv_v6_12301728"
MODEL="countex"
DATA_SPLIT="ALL"
WEEK="W3"

export WANDB_PROJECT="${PROJECT_NAME}"
export WANDB_NAME="${EXPERIMENT_NAME}"

accelerate launch --main_process_port 29500 --config_file ./ddp_cfgs/1n6r.yaml train.py \
    --backbone_size "${SIZE}" \
    --model "${MODEL}" \
    --save_qualitative_results False \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_epochs 10 \
    --learning_rate 1.5e-6 \
    --weight_decay 0.00001 \
    --warmup_steps 0 \
    --logging_steps 50 \
    --eval_steps 1000 \
    --save_steps 5000 \
    --dataloader_num_workers 4 \
    --output_dir "/nfs/130.245.4.102/add_disk0/yifeng/fg_count/fg_count_exp/${EXPERIMENT_NAME}" \
    --train_data_path "yifehuang97/CoCount-train-v6" \
    --val_data_path "BBVisual/CoCount-val" \
    --test_data_path "BBVisual/CoCount-test" \
    --weakly_supervised_data_path "BBVisual/CoCount-train" \
    --save_total_limit 10 \
    --remove_unused_columns False \
    --dataloader_pin_memory False \
    --bf16 True \
    --report_to "wandb" \
    --run_name "${EXPERIMENT_NAME}" \
    --lr_scheduler_type "cosine" \
    --use_weakly_supervised_training False \
    --seed 666 \
    --data_split "${DATA_SPLIT}" \
    --weakly_supervised_sample_num 1
