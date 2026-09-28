# coding=utf-8
"""
Build a size-matched RANDOM-selection training set (ACCV'26 rebuttal, PTWy MW2).

Table 5 row (a), "w/o Data Selection", is not a controlled ablation: relative to
the full model it changes five things at once (scripts/aba/wo_selection.sh vs
scripts/train/w8/countex_with_prior/kc_stage2.sh):
    training set   BBVisual/CoCount-train-2 (155,750 weak rows, ZERO dot-labelled)
                   vs 14,834 dot + 7,397 selected weak
    subsample      --train_data_ratio 0.5
    epochs         1  vs 3
    LR             5e-6 vs 1e-5
so it cannot support the claim that the gain comes from selection.

This builds the clean control: the SAME 14,834 dot rows plus 7,397 weak rows
drawn uniformly at random from the same 100,000-sample scored pool. Data
quantity, initialisation, optimiser and schedule are then identical to the full
model and only the *quality* of the weak rows changes.
"""
import os, json
os.environ.setdefault("HF_HOME", "${HF_HOME:-~/.cache/huggingface}")


import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict

CATS = ["FOO", "HOU", "FUN", "OFF", "OTR"]
SEED = 666
OUT = "${DATA_ROOT:-./data}/CoCount-WS-train-random-matched"
REF = "yifehuang97/CoCount-WS-train-v2-acc-selection-sem-100k-v2"


def main():
    ref = load_dataset(REF)
    rng = np.random.default_rng(SEED)
    out, summary = {}, {}
    for cat in CATS:
        base = ref[cat].filter(lambda r: r["type"] == "dot_anno", num_proc=8)
        n_weak = len(ref[cat]) - len(base)          # exactly what selection retained

        pool = load_dataset(f"yifehuang97/CoCount-train-2-100k-{cat}-pseudo-label")["train"]
        idx = sorted(rng.choice(len(pool), size=n_weak, replace=False).tolist())
        sel = pool.select(idx)

        keep = base.column_names
        sel = sel.remove_columns([c for c in sel.column_names if c not in keep])
        for col in ("positive_exemplars", "negative_exemplars"):
            if col not in sel.column_names:
                sel = sel.add_column(col, [None] * len(sel))
        sel = sel.map(lambda r: {"type": "dot_anno_pseudo"}, num_proc=8)
        sel = sel.select_columns(keep).cast(base.features)

        out[cat] = concatenate_datasets([base, sel])
        summary[cat] = {"dot": len(base), "weak_random": len(sel),
                        "weak_selected_in_ref": n_weak, "total": len(out[cat])}
        print(cat, summary[cat], flush=True)

    DatasetDict(out).save_to_disk(OUT)
    summary["TOTAL"] = {k: sum(summary[c][k] for c in CATS) for k in summary["FOO"]}
    print(json.dumps(summary, indent=2))
    with open("matched_eval/random_selection_dataset_summary.json", "w") as f:
        json.dump({"path": OUT, "seed": SEED, "counts": summary}, f, indent=2)
    print("Saved ->", OUT)


if __name__ == "__main__":
    main()
