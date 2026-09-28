# coding=utf-8
"""
Localization evaluation with dot annotations (ACCV'26 rebuttal, reviewer jWMT):
"Providing evaluation on object localisation using dot annotations will be helpful."

CoCount val/test carry no dots, but every *train* split does.  In the NC setting
the held-out supercategory's train split is never seen by either the pretrained
CountEx-NC-X model or by our stage-2 model, so it is a legitimate held-out set
with dot supervision available for scoring only.

Reported per checkpoint, for both inference heads:
  * GAME(L), L = 0..3  -- grid average mean error; GAME(0) is plain MAE and each
    higher L penalises count mass that lands in the wrong image region.
  * point precision / recall / F1 for the detection head, via Hungarian matching
    of predicted box centres to GT dots at a normalized distance tolerance.
"""

import os
import json
import time
import argparse

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
from scipy.optimize import linear_sum_assignment

from hf_model import CountEX
from hf_model.CountEXStage2 import CountEXStage2
from utils import collator

MODEL_MAP = {"countex": CountEX, "countex_stage2": CountEXStage2}
GAME_LEVELS = [0, 1, 2, 3]
TOLERANCES = [0.05, 0.10]


def grid_counts_from_points(pts_norm, L):
    """#points per cell of a 2^L x 2^L grid; pts_norm is (N,2) in [0,1] as (x,y)."""
    g = 2 ** L
    out = np.zeros((g, g), dtype=np.float64)
    if len(pts_norm) == 0:
        return out
    xs = np.clip((pts_norm[:, 0] * g).astype(int), 0, g - 1)
    ys = np.clip((pts_norm[:, 1] * g).astype(int), 0, g - 1)
    np.add.at(out, (ys, xs), 1.0)
    return out


def grid_sums_from_density(density, L):
    """Sum of a HxW density map over each cell of a 2^L x 2^L grid."""
    g = 2 ** L
    H, W = density.shape
    ys = np.linspace(0, H, g + 1).round().astype(int)
    xs = np.linspace(0, W, g + 1).round().astype(int)
    out = np.zeros((g, g), dtype=np.float64)
    for i in range(g):
        for j in range(g):
            out[i, j] = density[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].sum()
    return out


def point_prf(pred_pts, gt_pts, tol):
    """Hungarian-matched precision/recall/F1 at a normalized distance tolerance."""
    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return 0.0, 0.0, 0.0, 0
    d = np.linalg.norm(pred_pts[:, None, :] - gt_pts[None, :, :], axis=-1)
    cost = np.where(d <= tol, d, 1e6)
    r, c = linear_sum_assignment(cost)
    tp = int(((d[r, c] <= tol)).sum())
    prec = tp / len(pred_pts)
    rec = tp / len(gt_pts)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp


@torch.no_grad()
def run(model, dataset, device, threshold, prefix):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator())

    game_den = {L: [] for L in GAME_LEVELS}
    game_det = {L: [] for L in GAME_LEVELS}
    prf = {t: {"p": [], "r": [], "f": []} for t in TOLERANCES}
    n_gt, n_det = [], []
    t0 = time.time()

    for step, inputs in enumerate(tqdm(loader, desc=f"[{prefix}]")):
        pts = np.asarray(inputs["pos_points"][0], dtype=np.float64)
        if pts.size == 0:
            continue
        W, H = inputs["shapes"][0]
        pts_norm = np.stack([pts[:, 0] / W, pts[:, 1] / H], axis=1)
        pts_norm = np.clip(pts_norm, 0.0, 1.0 - 1e-9)

        pos = inputs["pos_llm_det_inputs"].to(device)
        pos["pixel_values"] = pos["pixel_values"].to(torch.bfloat16)
        if "pos_exemplars" in inputs:
            pos["pos_exemplars"] = inputs["pos_exemplars"]
        if "neg_exemplars" in inputs:
            pos["neg_exemplars"] = inputs["neg_exemplars"]
        neg = {k: v.to(device) for k, v in inputs["neg_llm_det_inputs"].items()}
        neg["pixel_values"] = neg["pixel_values"].to(torch.bfloat16)
        pos["neg_token_type_ids"] = neg["token_type_ids"]
        pos["neg_attention_mask"] = neg["attention_mask"]
        pos["neg_pixel_mask"] = neg["pixel_mask"]
        pos["neg_pixel_values"] = neg["pixel_values"]
        pos["neg_input_ids"] = neg["input_ids"]
        pos["use_neg"] = True

        out = model(**pos)

        if getattr(out, "pred_count", None) is not None:
            density = out["density_map"][0, 0].float().cpu().numpy()
        else:
            density = out["density_map_pred"][0, 0].float().cpu().numpy()

        scores = torch.sigmoid(out["logits"]).max(dim=-1)[0][0].float().cpu().numpy()
        boxes = out["pred_boxes"][0].float().cpu().numpy()  # cx cy w h, normalized
        keep = scores > threshold
        centers = boxes[keep][:, :2]
        centers = np.clip(centers, 0.0, 1.0 - 1e-9)

        n_gt.append(len(pts_norm))
        n_det.append(int(keep.sum()))

        for L in GAME_LEVELS:
            gt_g = grid_counts_from_points(pts_norm, L)
            game_den[L].append(np.abs(grid_sums_from_density(density, L) - gt_g).sum())
            game_det[L].append(np.abs(grid_counts_from_points(centers, L) - gt_g).sum())

        for t in TOLERANCES:
            p, r, f, _ = point_prf(centers, pts_norm, t)
            prf[t]["p"].append(p)
            prf[t]["r"].append(r)
            prf[t]["f"].append(f)

    res = {
        "n_images": len(n_gt),
        "mean_gt_dots": float(np.mean(n_gt)) if n_gt else None,
        "mean_detections": float(np.mean(n_det)) if n_det else None,
        "GAME_density": {f"GAME{L}": float(np.mean(game_den[L])) for L in GAME_LEVELS},
        "GAME_detection": {f"GAME{L}": float(np.mean(game_det[L])) for L in GAME_LEVELS},
        "point_matching": {
            f"tol_{t}": {
                "precision": float(np.mean(prf[t]["p"])),
                "recall": float(np.mean(prf[t]["r"])),
                "f1": float(np.mean(prf[t]["f"])),
            } for t in TOLERANCES
        },
        "threshold": threshold,
        "total_sec": time.time() - t0,
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="countex", choices=list(MODEL_MAP))
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--data_split", required=True, help="held-out supercategory, e.g. FOO")
    ap.add_argument("--train_data_path", default="BBVisual/CoCount-train")
    ap.add_argument("--bbx_threshold", type=float, default=0.42)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", default="./localization_eval")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset(args.train_data_path)[args.data_split]
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    model = MODEL_MAP[args.model].from_pretrained(args.ckpt_path).to(device).to(torch.bfloat16)
    model.eval()

    res = run(model, ds, device, args.bbx_threshold, f"{args.tag}/{args.data_split}")
    res.update(tag=args.tag, model=args.model, ckpt_path=args.ckpt_path,
               data_split=args.data_split, train_data_path=args.train_data_path)
    print(json.dumps(res, indent=2))
    with open(os.path.join(args.out_dir, f"{args.tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved -> {args.out_dir}/{args.tag}.json")


if __name__ == "__main__":
    main()
