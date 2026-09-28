#!/bin/bash
# Unified launcher for every ACCV'26-rebuttal training run.
#
# All variants share exactly the same recipe as the paper's KC stage-2 run
# (scripts/train/w8/countex_with_prior/kc_stage2.sh): same init, LR 1e-5 constant,
# 3 epochs, and effective batch size 6 -- here realised as 2 GPUs x grad_accum 3
# instead of the paper's 6 GPUs x grad_accum 1, so every run below is directly
# comparable to every other.
#
# Usage: VARIANT=<name> GPUS=0,1 PORT=29520 bash train_variant.sh
#
# Variants
#   full_seed666 / full_seed777 / full_seed888
#       the complete method; three seeds give the mean +/- std reviewer o9Vb asked for
#   ctrl_density_count_only
#       (qc2J/o9Vb control i) density head trained with L_count only
#   ctrl_decoder_count_only
#       (qc2J/o9Vb control ii) decoder trained with |sum sigmoid(l_n) - c| only
#   sel_acc_only
#       (qc2J) selection ablation: count agreement kept, semantic check removed
#   sel_random
#       (PTWy) size-matched no-selection control: the same 7,397 weak slots filled
#       with random pool samples, so only weak-label quality changes
set -eu

VARIANT="${VARIANT:?set VARIANT}"
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++
export TOKENIZERS_PARALLELISM="false"
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"   # optional, leave empty to disable W&B
export HF_TOKEN="${HF_TOKEN:?export your HuggingFace token first}"
export WANDB_DIR="${WANDB_DIR:-./wandb}"
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER}"

WEEK="ACCVRebuttal"
PROJECT_NAME="ACCV_Rebuttal"
SIZE="tiny"
MODEL="countex_stage2"
DATA_SPLIT="ALL"
PRETRAINED_PATH="BBVisual/CountEX-KC"
TRAIN_DATA="yifehuang97/CoCount-WS-train-v2-acc-selection-sem-100k-v2"

# --- defaults = the full method -------------------------------------------
SEED=666
USE_PSEUDO=True
PSEUDO_W=500.0
DENSITY_PHASE=100
DETECT_PHASE=10
ALIGN_W=0.005
DET_COUNT_W=0.005
USE_MT=True
CONSIST_W=500.0

case "$VARIANT" in
  full_seed666) SEED=666 ;;
  full_seed777) SEED=777 ;;
  full_seed888) SEED=888 ;;
  sel_acc_only)
    TRAIN_DATA="${DATA_ROOT:-./data}/CoCount-WS-train-acc-only-tau005" ;;
  sel_random)
    TRAIN_DATA="${DATA_ROOT:-./data}/CoCount-WS-train-random-matched" ;;
  ctrl_density_count_only)
    USE_PSEUDO=False; DENSITY_PHASE=100000000; DETECT_PHASE=0
    ALIGN_W=0.0; DET_COUNT_W=0.0; USE_MT=False ;;
  ctrl_decoder_count_only)
    USE_PSEUDO=False; DENSITY_PHASE=0; DETECT_PHASE=100000000
    ALIGN_W=0.0; DET_COUNT_W=0.005; USE_MT=False ;;
  *) echo "unknown VARIANT: $VARIANT" >&2; exit 1 ;;
esac

EXPERIMENT_NAME="rb_${VARIANT}"
OUT_DIR="${EXP_ROOT:-./experiments}/${WEEK}/${EXPERIMENT_NAME}"
export WANDB_PROJECT="${PROJECT_NAME}"
export WANDB_NAME="${EXPERIMENT_NAME}"

cd "$(dirname "$0")/../.."
echo "=== ${EXPERIMENT_NAME} on GPUs ${CUDA_VISIBLE_DEVICES} (seed ${SEED}) ==="
echo "    train_data=${TRAIN_DATA}"
echo "    pseudo=${USE_PSEUDO} density_phase=${DENSITY_PHASE} detect_phase=${DETECT_PHASE}"
echo "    align_w=${ALIGN_W} det_count_w=${DET_COUNT_W} mean_teacher=${USE_MT}"
date

ACCELERATE=accelerate

set +e
$ACCELERATE launch --main_process_port "${PORT:-29520}" --config_file ./ddp_cfgs/${DDP_CFG:-1n2r}.yaml train_alternating.py \
    --backbone_size "${SIZE}" \
    --model "${MODEL}" \
    --pretrained_path "${PRETRAINED_PATH}" \
    --save_qualitative_results False \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${ACCUM:-3} \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.00001 \
    --logging_steps 50 \
    --eval_steps 2000 \
    --save_steps 6000 \
    --dataloader_num_workers 4 \
    --output_dir "${OUT_DIR}" \
    --train_data_path "${TRAIN_DATA}" \
    --val_data_path "BBVisual/CoCount-val" \
    --test_data_path "BBVisual/CoCount-test" \
    --weakly_supervised_data_path "BBVisual/FG-Count-V2-weak-sup-train" \
    --save_total_limit 1 \
    --remove_unused_columns False \
    --dataloader_pin_memory False \
    --bf16 True \
    --save_safetensors False \
    --report_to "wandb" \
    --run_name "${EXPERIMENT_NAME}" \
    --wandb_name "${EXPERIMENT_NAME}" \
    --wandb_project "${PROJECT_NAME}" \
    --lr_scheduler_type "constant" \
    --seed "${SEED}" \
    --data_split "${DATA_SPLIT}" \
    --weakly_supervised_sample_num 1 \
    --use_weakly_supervised_training True \
    --use_uncertainty_loss False \
    --count_loss_type "interval_huber" \
    --count_loss_weight 1.0 \
    --use_neg_prob 0.0 \
    --use_pseudo_density_loss "${USE_PSEUDO}" \
    --pseudo_density_weight "${PSEUDO_W}" \
    --use_alternating_training True \
    --density_phase_length "${DENSITY_PHASE}" \
    --detection_phase_length "${DETECT_PHASE}" \
    --detection_alignment_weight "${ALIGN_W}" \
    --detection_count_loss_weight "${DET_COUNT_W}" \
    --use_mean_teacher "${USE_MT}" \
    --consistency_weight "${CONSIST_W}" \
    --ema_decay 0.999 \
    --save_best_det True \
    --max_grad_norm 1.0

RC=$?
set -e
if [ $RC -ne 0 ]; then echo "=== ${EXPERIMENT_NAME} FAILED rc=$RC ==="; exit $RC; fi
echo "=== ${EXPERIMENT_NAME} FINISHED ==="
date
