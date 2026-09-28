# coding=utf-8
"""
PairTally evaluation for the ACCV'26 rebuttal.

Extends eval_pairtally_stage2.py with everything the reviewers asked for:
  * both inference heads (density integral AND thresholded detection, plus a
    full detection-threshold sweep)            -> jWMT / qc2J matched inference
  * per-image dump + inter/intra breakdown by GT count bin and an outlier
    analysis                                   -> PTWy intra-class regression
  * dot-based localization (GAME 0-3, point P/R/F1), since PairTally ships dots
                                               -> jWMT "where is the spatial info"
"""

import os
import json
import time
import argparse

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import GroundingDinoProcessor
from scipy.optimize import linear_sum_assignment

from hf_model import CountEX, CountEXStage2

ANN = "${PAIRTALLY_ROOT:-./data/PairTally}"
IMG_ROOT = "${PAIRTALLY_ROOT:-./data/PairTally}"

MODEL_MAP = {"countex": CountEX, "countex_stage2": CountEXStage2}
THRESHOLDS = sorted(set([round(0.05 * k, 2) for k in range(1, 20)] + [0.42, 0.3]))
GAME_LEVELS = [0, 1, 2, 3]
TOLERANCES = [0.05, 0.10]
COUNT_BINS = [(0, 25), (26, 50), (51, 100), (101, 10 ** 9)]


def bin_of(c):
    for lo, hi in COUNT_BINS:
        if lo <= c <= hi:
            return f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"
    return "?"


def norm_boxes(exemplars, w, h):
    out = []
    for ex in exemplars:
        tlx, tly = ex[0][0] / w, ex[0][1] / h
        brx, bry = ex[2][0] / w, ex[2][1] / h
        tlx, tly = max(tlx, 0.0), max(tly, 0.0)
        tlx, tly = min(tlx, 1 - 1e-4), min(tly, 1 - 1e-4)
        brx, bry = min(brx, 1.0), min(bry, 1.0)
        brx, bry = max(brx, tlx), max(bry, tly)
        out.append([tlx, tly, brx, bry])
    return torch.from_numpy(np.array(out)).float()


def grid_counts(pts, L):
    g = 2 ** L
    out = np.zeros((g, g))
    if len(pts) == 0:
        return out
    xs = np.clip((pts[:, 0] * g).astype(int), 0, g - 1)
    ys = np.clip((pts[:, 1] * g).astype(int), 0, g - 1)
    np.add.at(out, (ys, xs), 1.0)
    return out


def grid_density(den, L):
    g = 2 ** L
    H, W = den.shape
    ys = np.linspace(0, H, g + 1).round().astype(int)
    xs = np.linspace(0, W, g + 1).round().astype(int)
    return np.array([[den[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].sum() for j in range(g)] for i in range(g)])


def point_f1(pred, gt, tol):
    if len(pred) == 0 or len(gt) == 0:
        return 0.0, 0.0, 0.0
    d = np.linalg.norm(pred[:, None, :] - gt[None, :, :], axis=-1)
    r, c = linear_sum_assignment(np.where(d <= tol, d, 1e6))
    tp = int((d[r, c] <= tol).sum())
    p, rec = tp / len(pred), tp / len(gt)
    return p, rec, (2 * p * rec / (p + rec) if p + rec > 0 else 0.0)


def agg(errs, gts):
    errs, gts = np.asarray(errs, float), np.asarray(gts, float)
    if errs.size == 0:
        return None
    return {"n": int(errs.size), "mae": float(errs.mean()),
            "rmse": float(np.sqrt((errs ** 2).mean())),
            "nae": float((errs / np.clip(gts, 1e-6, None)).mean())}


def summarize(records, key):
    """key is 'den' or 'det'; returns overall / inter / intra / per-count-bin metrics."""
    def sub(rs):
        return agg([abs(r[key] - r["gt"]) for r in rs], [r["gt"] for r in rs])
    out = {"overall": sub(records),
           "inter": sub([r for r in records if r["split"] == "INTER"]),
           "intra": sub([r for r in records if r["split"] == "INTRA"])}
    for split in ["INTER", "INTRA"]:
        for b in sorted({bin_of(r["gt"]) for r in records}):
            rs = [r for r in records if r["split"] == split and bin_of(r["gt"]) == b]
            if rs:
                out[f"{split}_count_{b}"] = sub(rs)
    # outlier diagnostic: contribution of the worst 5% of images to intra MAE
    intra = sorted([r for r in records if r["split"] == "INTRA"],
                   key=lambda r: abs(r[key] - r["gt"]), reverse=True)
    if intra:
        k = max(1, int(round(0.05 * len(intra))))
        worst = intra[:k]
        tot = sum(abs(r[key] - r["gt"]) for r in intra)
        out["intra_worst5pct"] = {
            "n": k,
            "mae_of_worst": float(np.mean([abs(r[key] - r["gt"]) for r in worst])),
            "share_of_intra_total_error": float(sum(abs(r[key] - r["gt"]) for r in worst) / tot) if tot > 0 else None,
            "image_names": [r["image"] for r in worst[:15]],
        }
        rest = intra[k:]
        out["intra_excl_worst5pct"] = agg([abs(r[key] - r["gt"]) for r in rest], [r["gt"] for r in rest])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="countex", choices=list(MODEL_MAP))
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--bbx_threshold", type=float, default=0.42)
    ap.add_argument("--out_dir", default="./pairtally_eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_localization", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = GroundingDinoProcessor.from_pretrained("fushh7/llmdet_swin_tiny_hf")
    model = MODEL_MAP[args.model].from_pretrained(args.ckpt_path).to(device).to(torch.bfloat16)
    model.eval()

    annotations = json.load(open(ANN))
    names = list(annotations.keys())
    if args.limit > 0:
        names = names[:args.limit]

    records = []
    game_den = {L: [] for L in GAME_LEVELS}
    game_det = {L: [] for L in GAME_LEVELS}
    prf = {t: [] for t in TOLERANCES}
    t0 = time.time()

    for name in tqdm(names, desc=f"[pairtally/{args.tag}]"):
        ann = annotations[name]
        img = Image.open(os.path.join(IMG_ROOT, name)).convert("RGB")
        w, h = img.size
        pos_cap = ann["positive_prompt"] + "."
        neg_cap = ann["negative_prompt"] + "."

        pos = proc(images=img, text=pos_cap, return_tensors="pt", padding=True).to(device)
        pos["pixel_values"] = pos["pixel_values"].to(torch.bfloat16)
        neg = proc(images=img, text=neg_cap, return_tensors="pt", padding=True)
        neg = {k: v.to(device) for k, v in neg.items()}
        neg["pixel_values"] = neg["pixel_values"].to(torch.bfloat16)
        pos["pos_exemplars"] = norm_boxes(ann["box_examples_coordinates"], w, h)
        pos["neg_exemplars"] = norm_boxes(ann["negative_box_exemples_coordinates"], w, h)
        pos["neg_token_type_ids"] = neg["token_type_ids"]
        pos["neg_attention_mask"] = neg["attention_mask"]
        pos["neg_pixel_mask"] = neg["pixel_mask"]
        pos["neg_pixel_values"] = neg["pixel_values"]
        pos["neg_input_ids"] = neg["input_ids"]
        pos["use_neg"] = True

        with torch.no_grad():
            out = model(**pos)

        if getattr(out, "pred_count", None) is not None:
            den_map = out["density_map"][0, 0].float().cpu().numpy()
            den_count = float(out["pred_count"].float().item())
        else:
            den_map = out["density_map_pred"][0, 0].float().cpu().numpy()
            den_count = float(den_map.sum())

        scores = torch.sigmoid(out["logits"]).max(dim=-1)[0][0].float().cpu().numpy()
        boxes = out["pred_boxes"][0].float().cpu().numpy()
        det_count = int((scores > args.bbx_threshold).sum())
        sweep = [int((scores > t).sum()) for t in THRESHOLDS]

        gt_pts = np.asarray(ann["points"], dtype=np.float64)
        gt = float(len(gt_pts))
        records.append({"image": name, "gt": gt, "den": den_count, "det": det_count,
                        "det_sweep": sweep,
                        "split": "INTER" if "INTER" in name else "INTRA",
                        "category": name.split("_")[0]})

        if not args.skip_localization and len(gt_pts) > 0:
            gtn = np.clip(np.stack([gt_pts[:, 0] / w, gt_pts[:, 1] / h], 1), 0, 1 - 1e-9)
            centers = np.clip(boxes[scores > args.bbx_threshold][:, :2], 0, 1 - 1e-9)
            for L in GAME_LEVELS:
                g = grid_counts(gtn, L)
                game_den[L].append(np.abs(grid_density(den_map, L) - g).sum())
                game_det[L].append(np.abs(grid_counts(centers, L) - g).sum())
            for t in TOLERANCES:
                prf[t].append(point_f1(centers, gtn, t))

    gts = [r["gt"] for r in records]
    sweep_metrics = {}
    for j, t in enumerate(THRESHOLDS):
        sweep_metrics[str(t)] = agg([abs(r["det_sweep"][j] - r["gt"]) for r in records], gts)
    best_t = min(sweep_metrics, key=lambda t: sweep_metrics[t]["mae"])

    result = {
        "tag": args.tag, "model": args.model, "ckpt_path": args.ckpt_path,
        "bbx_threshold": args.bbx_threshold, "n_images": len(records),
        "density": summarize(records, "den"),
        "detection": summarize(records, "det"),
        "detection_sweep_overall": sweep_metrics,
        "detection_best_threshold": float(best_t),
        "detection_best_overall": sweep_metrics[best_t],
        "sec_per_image": (time.time() - t0) / max(len(records), 1),
    }
    if not args.skip_localization and game_den[0]:
        result["localization"] = {
            "GAME_density": {f"GAME{L}": float(np.mean(game_den[L])) for L in GAME_LEVELS},
            "GAME_detection": {f"GAME{L}": float(np.mean(game_det[L])) for L in GAME_LEVELS},
            "point_matching": {
                f"tol_{t}": {
                    "precision": float(np.mean([x[0] for x in prf[t]])),
                    "recall": float(np.mean([x[1] for x in prf[t]])),
                    "f1": float(np.mean([x[2] for x in prf[t]])),
                } for t in TOLERANCES},
        }

    with open(os.path.join(args.out_dir, f"{args.tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.out_dir, f"{args.tag}_preds.json"), "w") as f:
        json.dump(records, f)
    print(json.dumps({k: v for k, v in result.items() if k != "detection_sweep_overall"}, indent=2))
    print(f"Saved -> {args.out_dir}/{args.tag}.json")


if __name__ == "__main__":
    main()
