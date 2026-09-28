# coding=utf-8
"""
Build the count-agreement-ONLY selected training set (ACCV'26 rebuttal, qc2J:
"consider separately ablating the count-agreement and semantic-consistency
selection criteria").

The released main training set (CoCount-WS-train-v2-acc-selection-sem-100k-v2)
is: 14,834 dot-labelled samples  +  weak samples passing BOTH
    count agreement  |c - #pseudo points| <= int(0.05 * c)   (pos and neg)
    semantic check   sup_caption == 'NA.' or #sup_points >= pos_count
which retains 7,397 of the 100,000-sample pool.

This script rebuilds the same set with the semantic check REMOVED, retaining
9,626 weak samples, and writes it to disk so it can be trained on directly.
"""
import os, json
os.environ.setdefault("HF_HOME", "${HF_HOME:-~/.cache/huggingface}")


import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict

CATS = ["FOO", "HOU", "FUN", "OFF", "OTR"]
TAU = 0.05
OUT = "${DATA_ROOT:-./data}/CoCount-WS-train-acc-only-tau005"

REF = "yifehuang97/CoCount-WS-train-v2-acc-selection-sem-100k-v2"


def main():
    ref = load_dataset(REF)
    out = {}
    summary = {}
    for cat in CATS:
        # dot-labelled part of the released set: identical schema, identical rows
        base = ref[cat].filter(lambda r: r["type"] == "dot_anno", num_proc=8)

        pool = load_dataset(f"yifehuang97/CoCount-train-2-100k-{cat}-pseudo-label")["train"]
        meta = pool.remove_columns([c for c in pool.column_names
                                    if c not in ("pos_count", "neg_count", "pos_points", "neg_points")]).to_dict()
        pc = np.asarray(meta["pos_count"]); nc = np.asarray(meta["neg_count"])
        pdisp = np.asarray([abs(a - len(b)) for a, b in zip(meta["pos_count"], meta["pos_points"])])
        ndisp = np.asarray([abs(a - len(b)) for a, b in zip(meta["neg_count"], meta["neg_points"])])
        acc = (pdisp <= (pc * TAU).astype(int)) & (ndisp <= (nc * TAU).astype(int))
        idx = np.where(acc)[0].tolist()

        sel = pool.select(idx)
        # align schema with the released set: mark type and add the (unused)
        # exemplar columns, which the pseudo-labelled pool does not carry.
        keep = base.column_names
        sel = sel.remove_columns([c for c in sel.column_names if c not in keep])
        for col in ("positive_exemplars", "negative_exemplars"):
            if col not in sel.column_names:
                sel = sel.add_column(col, [None] * len(sel))
        sel = sel.map(lambda r: {"type": "dot_anno_pseudo"}, num_proc=8)
        missing = [c for c in keep if c not in sel.column_names]
        assert not missing, f"{cat}: pseudo split is missing columns {missing}"
        sel = sel.select_columns(keep).cast(base.features)

        out[cat] = concatenate_datasets([base, sel])
        summary[cat] = {"dot": len(base), "weak_acc_only": len(sel),
                        "weak_acc_and_sem_released": len(ref[cat]) - len(base),
                        "total": len(out[cat])}
        print(cat, summary[cat], flush=True)

    dd = DatasetDict(out)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    dd.save_to_disk(OUT)
    summary["TOTAL"] = {k: sum(summary[c][k] for c in CATS) for k in summary["FOO"]}
    print(json.dumps(summary, indent=2))
    with open("matched_eval/acc_only_dataset_summary.json", "w") as f:
        json.dump({"path": OUT, "tau": TAU, "counts": summary}, f, indent=2)
    print("Saved ->", OUT)


if __name__ == "__main__":
    main()
