# coding=utf-8
"""Clean inference-latency benchmark on one idle GPU (o9Vb: computational overhead).

Three configurations, same images, same GPU, back to back:
  countex           - the pretrained baseline
  ours              - our checkpoint exactly as shipped
  ours_no_prior     - our checkpoint with the training-only detection prior
                      skipped (monkey-patched), i.e. what a deployed model does
"""
import os, json, time, argparse
os.environ.setdefault("HF_HOME", "${HF_HOME:-~/.cache/huggingface}")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

from hf_model import CountEX
from hf_model.CountEXStage2 import CountEXStage2
from utils import collator

CATS = ["FOO", "FUN", "OFF", "OTR", "HOU"]


def prep(inputs, device):
    """Build a fresh kwargs dict. The cached batches are reused across configs,
    so this must not mutate `inputs` -- BatchEncoding.to() returns the same
    object, and a stashed `use_neg` bool would break the next .to(device)."""
    src = inputs["pos_llm_det_inputs"]
    pos = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in src.items()}
    pos["pixel_values"] = pos["pixel_values"].to(torch.bfloat16)
    for k in ("pos_exemplars", "neg_exemplars"):
        if k in inputs:
            pos[k] = inputs[k]
    neg = {k: (v.to(device) if torch.is_tensor(v) else v)
           for k, v in inputs["neg_llm_det_inputs"].items()}
    neg["pixel_values"] = neg["pixel_values"].to(torch.bfloat16)
    pos["neg_token_type_ids"] = neg["token_type_ids"]
    pos["neg_attention_mask"] = neg["attention_mask"]
    pos["neg_pixel_mask"] = neg["pixel_mask"]
    pos["neg_pixel_values"] = neg["pixel_values"]
    pos["neg_input_ids"] = neg["input_ids"]
    pos["use_neg"] = True
    return pos


@torch.no_grad()
def bench(model, batches, device, warmup=10):
    model.eval()
    for b in batches[:warmup]:
        model(**prep(b, device))
    torch.cuda.synchronize()
    ts = []
    for b in batches:
        torch.cuda.synchronize(); t = time.perf_counter()
        model(**prep(b, device))
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    return {"n": len(ts), "mean_ms": float(ts.mean() * 1000),
            "median_ms": float(np.median(ts) * 1000), "std_ms": float(ts.std() * 1000)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours_ckpt", default="${EXP_ROOT:-./experiments}"
                                           "fg_count_exp/W8/kc_countex_stage2_100k_666/best_val_model")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default="matched_eval/latency.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    ds = concatenate_datasets([load_dataset("BBVisual/CoCount-val")[c] for c in CATS]).select(range(args.n))
    batches = list(DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator()))

    res = {}
    m = CountEX.from_pretrained("BBVisual/CountEX-KC").to(device).to(torch.bfloat16)
    res["countex"] = bench(m, batches, device); del m; torch.cuda.empty_cache()
    print("countex", res["countex"])

    m = CountEXStage2.from_pretrained(args.ours_ckpt).to(device).to(torch.bfloat16)
    res["ours"] = bench(m, batches, device)
    print("ours", res["ours"])

    # skip the training-only detection prior: patch the name CountEXStage2.forward
    # actually calls (the package re-exports a *class* of the same name, so the
    # module attribute is the only reliable handle).
    import importlib
    S2 = importlib.import_module("hf_model.CountEXStage2")
    orig = S2.create_soft_density_from_logits
    S2.create_soft_density_from_logits = lambda logits, boxes, height, width, temperature=1.0, sigma=3.0: \
        torch.zeros((logits.shape[0], 1, height, width), device=logits.device, dtype=torch.float32)
    res["ours_no_prior"] = bench(m, batches, device)
    S2.create_soft_density_from_logits = orig
    print("ours_no_prior", res["ours_no_prior"])

    res["gpu"] = torch.cuda.get_device_name(0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
