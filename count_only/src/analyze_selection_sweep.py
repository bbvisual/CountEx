# coding=utf-8
"""Identify which count-agreement threshold reproduces each released selected set,
and report how many samples count-agreement alone vs. +semantic-consistency keep."""
import os, json, collections
os.environ.setdefault("HF_HOME", "${HF_HOME:-~/.cache/huggingface}")


import numpy as np
from datasets import load_dataset

CATS = ["FOO", "HOU", "FUN", "OFF", "OTR"]
COLS = ["pos_count", "neg_count", "pos_points", "neg_points", "sup_caption", "sup_points", "category"]
TAUS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]
BINS = [(0, 10), (11, 25), (26, 50), (51, 100), (101, 10 ** 9)]


def bin_of(c):
    for lo, hi in BINS:
        if lo <= c <= hi:
            return f"{lo}-{hi if hi < 10 ** 9 else 'inf'}"
    return "?"


def main():
    per_cat = {}
    for cat in CATS:
        ds = load_dataset(f"yifehuang97/CoCount-train-2-100k-{cat}-pseudo-label")["train"]
        ds = ds.remove_columns([c for c in ds.column_names if c not in COLS])
        d = ds.to_dict()
        n = len(d["pos_count"])
        pc = np.asarray(d["pos_count"], dtype=np.int64)
        nc = np.asarray(d["neg_count"], dtype=np.int64)
        pdisp = np.asarray([abs(a - len(b)) for a, b in zip(d["pos_count"], d["pos_points"])])
        ndisp = np.asarray([abs(a - len(b)) for a, b in zip(d["neg_count"], d["neg_points"])])
        sup_na = np.asarray([c == "NA." for c in d["sup_caption"]])
        sup_cnt = np.asarray([len(p) for p in d["sup_points"]])
        sem = sup_na | (sup_cnt >= pc)

        entry = {"pool": int(n), "sem_kept": int(sem.sum()), "taus": {}}
        for tau in TAUS:
            acc = (pdisp <= (pc * tau).astype(int)) & (ndisp <= (nc * tau).astype(int))
            both = acc & sem
            entry["taus"][str(tau)] = {
                "acc_kept": int(acc.sum()),
                "acc_and_sem_kept": int(both.sum()),
                "sem_rejects_of_acc": int((acc & ~sem).sum()),
                "mean_count_acc_sem": float(pc[both].mean()) if both.any() else None,
                "hist_acc_sem": dict(collections.Counter(bin_of(c) for c in pc[both])),
                "hist_filtered": dict(collections.Counter(bin_of(c) for c in pc[~both])),
            }
        entry["hist_pool"] = dict(collections.Counter(bin_of(c) for c in pc))
        entry["mean_count_pool"] = float(pc.mean())
        per_cat[cat] = entry
        print(cat, {t: entry["taus"][t]["acc_and_sem_kept"] for t in entry["taus"]})

    total = {"pool": sum(per_cat[c]["pool"] for c in CATS),
             "sem_kept": sum(per_cat[c]["sem_kept"] for c in CATS), "taus": {}}
    for tau in TAUS:
        t = str(tau)
        total["taus"][t] = {k: sum(per_cat[c]["taus"][t][k] for c in CATS)
                            for k in ["acc_kept", "acc_and_sem_kept", "sem_rejects_of_acc"]}
    print("\nTOTAL", json.dumps(total["taus"], indent=2))

    out = {"per_category": per_cat, "total": total,
           "released_selected_set_sizes": {
               "CoCount-WS-train-v2-acc-selection-sem-100k-v2": 7397,
               "CoCount-WS-train-v2-acc-selection-sem-100k-02": 38741,
               "CoCount-WS-train-v2-acc-selection-sem-100k-03": 50849,
           },
           "dot_labeled_base_total": 14834,
           "weak_pool_total": 155750}
    os.makedirs("matched_eval", exist_ok=True)
    with open("matched_eval/selection_sweep.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved -> matched_eval/selection_sweep.json")


if __name__ == "__main__":
    main()
