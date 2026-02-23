#!/bin/bash
# export CC=$CONDA_PREFIX/bin/gcc
# export CXX=$CONDA_PREFIX/bin/g++
export HF_TOKEN="" # replace with your own token
export TOKENIZERS_PARALLELISM="false"
export CUDA_VISIBLE_DEVICES="1"

SIZE="tiny"
DATA_SPLIT="ALL"
MODEL="countex"
# eval
python eval.py \
    --ckpt_path "BBVisual/CountEX-KC" \
    --train_data_path "BBVisual/CoCount-train" \
    --val_data_path "BBVisual/CoCount-val" \
    --test_data_path "BBVisual/CoCount-test" \
    --weakly_supervised_data_path "BBVisual/CoCount-train" \
    --backbone_size "${SIZE}" \
    --output_dir "./kc_eval" \
    --data_split "${DATA_SPLIT}" \
    --model "${MODEL}"
