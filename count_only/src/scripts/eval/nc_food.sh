#!/bin/bash
# export CC=$CONDA_PREFIX/bin/gcc
# export CXX=$CONDA_PREFIX/bin/g++
export HF_TOKEN="${HF_TOKEN:?export your HuggingFace token first}" # replace with your own token
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="0"

SIZE="tiny"
DATA_SPLIT="FOO"
MODEL="countex"
# eval
python "$(dirname "$0")/../.."/rebuttal_eval_wo_processing.py \
    --ckpt_path "BBVisual/CountEX-NC-Food" \
    --train_data_path "BBVisual/CoCount-train" \
    --val_data_path "BBVisual/CoCount-val" \
    --test_data_path "BBVisual/CoCount-test" \
    --weakly_supervised_data_path "BBVisual/CoCount-train" \
    --backbone_size "${SIZE}" \
    --output_dir "./kc_eval" \
    --data_split "${DATA_SPLIT}" \
    --model "${MODEL}"