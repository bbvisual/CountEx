# coding=utf-8
"""
Matched-inference evaluation: for a single checkpoint and a single forward pass,
report BOTH counting modes:
  * detection : number of decoder boxes with score > bbx_threshold
  * density   : integral (sum) of the predicted density map

This gives the 2x2 {CountEx, Ours} x {detection, density} table requested by the
ACCV'26 reviewers (jWMT / qc2J) without any retraining.

Usage:
  python eval_matched.py --model countex        --ckpt_path BBVisual/CountEX-KC \
      --data_split ALL --tag countex_kc
  python eval_matched.py --model countex_stage2 --ckpt_path <exp>/best_val_model \
      --data_split ALL --tag ours_kc
"""

import os
import json
import math
import time
import argparse

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets

from hf_model import CountEX
from hf_model.CountEXStage2 import CountEXStage2
from utils import collator

CATEGORIES = ["FOO", "FUN", "OFF", "OTR", "HOU"]

# Detection confidence thresholds swept at evaluation time. The paper protocol
# uses 0.42; the sweep shows whether the gain is a thresholding artifact.
THRESHOLDS = [round(0.05 * k, 2) for k in range(1, 20)] + [0.42, 0.3]
THRESHOLDS = sorted(set(THRESHOLDS))

MODEL_MAP = {
    "countex": CountEX,
    "countex_stage2": CountEXStage2,
}


def load_split(path, data_split):
    ds = load_dataset(path)
    if data_split.upper() in ("ALL", "KC"):
        return concatenate_datasets([ds[c] for c in CATEGORIES])
    return ds[data_split.upper()]


def metrics_from(errs, gts):
    errs = np.asarray(errs, dtype=np.float64)
    gts = np.asarray(gts, dtype=np.float64)
    out = {
        "n": int(errs.size),
        "mae": float(errs.mean()),
        "rmse": float(np.sqrt((errs ** 2).mean())),
        "nae": float((errs / np.clip(gts, 1e-6, None)).mean()),
    }
    return out


def score_records(records, gts, bbx_threshold):
    """Score already-computed per-image predictions against a set of GT counts."""
    det_err = [abs(r["det"] - g) for r, g in zip(records, gts)]
    den_err = [abs(r["den"] - g) for r, g in zip(records, gts)]
    sweep = {}
    for j, t in enumerate(THRESHOLDS):
        errs = [abs(r["det_sweep"][j] - g) for r, g in zip(records, gts)]
        sweep[str(t)] = metrics_from(errs, gts)
    best_t = min(sweep, key=lambda t: sweep[t]["mae"])
    return {
        "detection": metrics_from(det_err, gts),
        "density": metrics_from(den_err, gts),
        "detection_sweep": sweep,
        "detection_best_threshold": float(best_t),
        "detection_best": sweep[best_t],
        "bbx_threshold": bbx_threshold,
    }


@torch.no_grad()
def evaluate(model, dataset, device, bbx_threshold, prefix, dump_path=None):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator())

    det_err, den_err, gts = [], [], []
    sweep_err = [[] for _ in THRESHOLDS]
    records = []
    t0 = time.time()
    pbar = tqdm(loader, desc=f"[{prefix}]")
    for step, inputs in enumerate(pbar):
        pos = inputs["pos_llm_det_inputs"].to(device)
        pos["pixel_values"] = pos["pixel_values"].to(torch.bfloat16)
        if "pos_exemplars" in inputs:
            pos["pos_exemplars"] = inputs["pos_exemplars"]
        if "neg_exemplars" in inputs:
            pos["neg_exemplars"] = inputs["neg_exemplars"]

        neg = inputs["neg_llm_det_inputs"]
        neg = {k: v.to(device) for k, v in neg.items()}
        neg["pixel_values"] = neg["pixel_values"].to(torch.bfloat16)
        pos["neg_token_type_ids"] = neg["token_type_ids"]
        pos["neg_attention_mask"] = neg["attention_mask"]
        pos["neg_pixel_mask"] = neg["pixel_mask"]
        pos["neg_pixel_values"] = neg["pixel_values"]
        pos["neg_input_ids"] = neg["input_ids"]
        pos["use_neg"] = True

        outputs = model(**pos)

        # --- detection count (at the protocol threshold + a full sweep) ---
        logits = outputs["logits"]
        scores = torch.sigmoid(logits).max(dim=-1)[0][0].float()  # [N]
        det_count = int((scores > bbx_threshold).sum().item())
        sweep_counts = [int((scores > t).sum().item()) for t in THRESHOLDS]

        # --- density count ---
        if getattr(outputs, "pred_count", None) is not None:
            den_count = float(outputs["pred_count"].float().item())
        else:
            den_count = float(outputs["density_map_pred"].float().sum().item())

        gt = float(inputs["pos_count"])
        det_err.append(abs(det_count - gt))
        den_err.append(abs(den_count - gt))
        gts.append(gt)
        for j, c in enumerate(sweep_counts):
            sweep_err[j].append(abs(c - gt))

        records.append({
            "step": step,
            "gt": gt,
            "det": det_count,
            "den": den_count,
            "det_sweep": sweep_counts,
            "category": inputs.get("category"),
            "pos_caption": inputs["pos_caption"],
            "neg_caption": inputs["neg_caption"],
        })

        pbar.set_postfix({
            "detMAE": f"{np.mean(det_err):.2f}",
            "denMAE": f"{np.mean(den_err):.2f}",
        })

    elapsed = time.time() - t0
    res = score_records(records, gts, bbx_threshold)
    res["sec_per_image"] = elapsed / max(len(det_err), 1)
    res["total_sec"] = elapsed
    if dump_path:
        with open(dump_path, "w") as f:
            json.dump(records, f)
    return res, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="countex", choices=list(MODEL_MAP))
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--data_split", default="ALL")
    ap.add_argument("--val_data_path", default="BBVisual/CoCount-val")
    ap.add_argument("--test_data_path", default="BBVisual/CoCount-test")
    ap.add_argument("--test_corr_data_path", default="BBVisual/CoCount-test-corrected0621")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--bbx_threshold", type=float, default=0.42)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", default="./matched_eval")
    ap.add_argument("--dump_predictions", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: only evaluate first N samples")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MODEL_MAP[args.model].from_pretrained(args.ckpt_path)
    model = model.to(device).to(torch.bfloat16)
    model.eval()

    result = {
        "tag": args.tag,
        "model": args.model,
        "ckpt_path": args.ckpt_path,
        "data_split": args.data_split,
        "bbx_threshold": args.bbx_threshold,
        "val_data_path": args.val_data_path,
        "test_data_path": args.test_data_path,
        "test_corr_data_path": args.test_corr_data_path,
    }

    requested = [s.strip() for s in args.splits.split(",") if s.strip()]
    # test and test_corr share the same images and only differ in the count labels,
    # so a single forward pass over the test images scores both.
    forward_splits = [s for s in requested if s != "test_corr"]
    if "test_corr" in requested and "test" not in forward_splits:
        forward_splits.append("test")

    test_records = None
    for split in forward_splits:
        path = {"val": args.val_data_path, "test": args.test_data_path}[split]
        ds = load_split(path, args.data_split)
        if args.limit > 0:
            ds = ds.select(range(min(args.limit, len(ds))))
        dump = os.path.join(args.out_dir, f"{args.tag}_{split}_preds.json") if args.dump_predictions else None
        res, records = evaluate(model, ds, device, args.bbx_threshold,
                                f"{args.tag}/{split}", dump)
        if split in requested:
            result[split] = res
        if split == "test":
            test_records = records
        print(f"\n=== {args.tag} / {split} ===")
        print(json.dumps({k: v for k, v in res.items() if k != "detection_sweep"}, indent=2))

    if "test_corr" in requested and test_records is not None:
        ds_c = load_split(args.test_corr_data_path, args.data_split)
        if args.limit > 0:
            ds_c = ds_c.select(range(min(args.limit, len(ds_c))))
        gts_c = [float(c) for c in ds_c["pos_count"]]
        assert len(gts_c) == len(test_records), (
            f"corrected test split has {len(gts_c)} rows but {len(test_records)} predictions")
        n_changed = sum(1 for r, g in zip(test_records, gts_c) if abs(r["gt"] - g) > 1e-6)
        res_c = score_records(test_records, gts_c, args.bbx_threshold)
        res_c["n_labels_changed_vs_original_test"] = n_changed
        result["test_corr"] = res_c
        print(f"\n=== {args.tag} / test_corr ({n_changed} labels differ from CoCount-test) ===")
        print(json.dumps({k: v for k, v in res_c.items() if k != "detection_sweep"}, indent=2))

    out_file = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_file}")


if __name__ == "__main__":
    main()
