<div align="center">

# Boosting Fine-Grained Visual Counting Using Count-Only Annotations

Yifeng Huang<sup>1</sup> · Minh Hoai<sup>2</sup>

<sup>1</sup>Department of Computer Science, Stony Brook University<br>
<sup>2</sup>Australian Institute for Machine Learning, Adelaide University

**ACCV 2026**

</div>

---

This directory adapts a pretrained [CountEx](../README.md) counter using **count-only** labels,
that is, global object counts with no dot or box annotations. Confident detections supply pseudo
spatial supervision to the density head, the density map recalibrates detection confidences, and
the two alternate under count-level constraints, with data selection and a Mean Teacher for
stability.

The result is a counter that improves over its fully supervised starting point on both the
known-category and novel-category splits of CoCount, without a single new dot annotation.

> **Which head produces the count.** Counts here are read from the **density integral**, not from
> thresholded detections. The pretrained CountEx baseline is strongest through its *detection*
> head, so any comparison has to say which head it uses. `eval_matched.py` reports both from a
> single forward pass, which is the recommended way to compare.

---

## Setup

Identical to the parent repository, see [Setup](../README.md#setup). Additionally:

```bash
pip install -r count_only/requirements.txt
```

`numpy` must stay below 2.0, since transformers 4.42 and torch 2.3 are built against 1.x.

No credentials are stored here. Export your own:

```bash
export HF_TOKEN=...              # required, for the model and dataset hubs
export WANDB_API_KEY=...         # optional, omit to disable W&B logging
export EXP_ROOT=./experiments    # where checkpoints are written
export DATA_ROOT=./data          # local datasets, if any
```

---

## Evaluation

```bash
cd count_only/src

# both heads plus a detection-threshold sweep
python eval_matched.py --model countex --ckpt_path BBVisual/CountEX-KC \
    --data_split ALL --tag countex_kc

python eval_matched.py --model countex_stage2 \
    --ckpt_path $EXP_ROOT/<run>/best_val_model --data_split ALL --tag ours_kc
```

`--data_split ALL` is the known-category (KC) setting. Pass one of `FOO FUN OFF OTR HOU` to hold
that supercategory out (NC setting).

| script | what it measures |
|---|---|
| `eval_matched.py` | both heads, plus a 19-point detection-threshold sweep |
| `eval_density.py` | count from the density integral only |
| `eval.py` | count from thresholded detections only |
| `eval_localization.py` | GAME(0-3) and point P/R/F1 against dot annotations |
| `eval_pairtally_matched.py` | PairTally, both heads, inter/intra breakdown |
| `bench_latency.py` | per-image latency, with and without the training-only prior |

---

## Training

```bash
cd count_only/src
bash scripts/train/kc_stage2.sh          # known-category
bash scripts/train/nc_food_stage2.sh     # hold out Food, and so on
```

Three epochs at lr 1e-5 from `BBVisual/CountEX-KC`, effective batch size 6. The provided scripts
use 6 GPUs with `grad_accum 1`. With fewer GPUs, keep the product constant, for example 3 GPUs
with `grad_accum 2`.

Two things that will bite you if you change the parallel setup:

- The DeepSpeed ZeRO-2 configs in `ddp_cfgs/` do **not** support gradient accumulation across
  processes here, and fail with `no_sync ... incompatible with gradient partitioning logic of ZeRO
  stage 2`. For `grad_accum > 1`, use the plain-DDP configs (`*_ddp.yaml`).
- Under DDP, pass `--save_safetensors False`. The model ties `class_embed` weights between the
  decoder and the top level, and safetensors refuses to serialize shared tensors.

---

## Ablations

```bash
bash scripts/ablation/component/kc_wo_detection_phase.sh   # no density-to-decoder feedback
bash scripts/ablation/component/kc_wo_mean_teacher.sh

VARIANT=ctrl_density_count_only GPUS=0,1,2 DDP_CFG=1n3r_ddp ACCUM=2 \
    bash scripts/ablation/train_variant.sh
```

`train_variant.sh` also provides `ctrl_decoder_count_only`, `sel_acc_only` and `sel_random`.

The data-selection helpers reproduce the training sets exactly. Count agreement alone keeps 9,626
of the 100,000 scored count-only images, and adding the semantic-consistency check leaves 7,397:

```bash
python analyze_selection_sweep.py              # retained counts over thresholds
python build_acc_only_dataset.py               # count agreement only
python build_random_selection_dataset.py       # size-matched random control
```

---

## Citation

```bibtex
@inproceedings{huang2026countonly,
  title     = {Boosting Fine-Grained Visual Counting Using Count-Only Annotations},
  author    = {Huang, Yifeng and Hoai, Minh},
  booktitle = {Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```
