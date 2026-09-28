# coding=utf-8
"""Aggregate every rebuttal evaluation into console tables + LaTeX fragments."""
import os, json, glob
import numpy as np

MATCHED = "matched_eval"
OUT = "matched_eval/rebuttal_tables.txt"

NICE = {
    "countex_kc": ("CountEx", "KC"),
    "ours_kc": ("Ours", "KC"),
    "ours_kc_seed02": ("Ours (tau=0.20 selection, 38.7k weak)", "KC"),
    "ours_kc_seed03": ("Ours (tau=0.30 selection, 50.8k weak)", "KC"),
    "ours_kc_wo_selection": ("Ours w/o selection", "KC"),
    "ours_kc_wo_detection_phase": ("Ours w/o detection phase", "KC"),
    "ours_kc_wo_mean_teacher": ("Ours w/o Mean Teacher", "KC"),
    "countex_nc_food": ("CountEx", "NC-Food"), "ours_nc_food": ("Ours", "NC-Food"),
    "countex_nc_home": ("CountEx", "NC-Home"), "ours_nc_home": ("Ours", "NC-Home"),
    "countex_nc_desk": ("CountEx", "NC-Desk"), "ours_nc_desk": ("Ours", "NC-Desk"),
    "countex_nc_misc": ("CountEx", "NC-Misc"), "ours_nc_misc": ("Ours", "NC-Misc"),
    "countex_nc_game": ("CountEx", "NC-Game"), "ours_nc_game": ("Ours", "NC-Game"),
    "ctrl_density_count_only": ("Control (i): density, L_count only", "KC"),
    "ctrl_decoder_count_only": ("Control (ii): decoder, soft-count only", "KC"),
    "sel_acc_only": ("Selection: count agreement only", "KC"),
    "full_seed666": ("Ours (repro, seed 666)", "KC"),
    "full_seed777": ("Ours (seed 777)", "KC"),
    "full_seed888": ("Ours (seed 888)", "KC"),
}

lines = []
def P(s=""):
    print(s)
    lines.append(s)


def load_all():
    out = {}
    for f in sorted(glob.glob(os.path.join(MATCHED, "*.json"))):
        tag = os.path.basename(f)[:-5]
        if tag.endswith("_preds") or tag.startswith("selection") or tag.startswith("acc_only"):
            continue
        try:
            out[tag] = json.load(open(f))
        except Exception as e:
            P(f"  [skip {tag}: {e}]")
    return out


def row(d, split, head):
    if split not in d:
        return None
    return d[split][head]


def main():
    res = load_all()
    P("=" * 100)
    P("MATCHED-INFERENCE TABLE  (MAE / RMSE; det@0.42 = paper protocol, det* = best sweep threshold)")
    P("=" * 100)
    hdr = f"{'run':<44}{'split':<10}{'density':>16}{'det@0.42':>16}{'det* (thr)':>22}"
    for split in ["val", "test", "test_corr"]:
        P(f"\n--- {split} ---")
        P(hdr)
        for tag, d in res.items():
            if split not in d:
                continue
            name, setting = NICE.get(tag, (tag, "?"))
            den, det = d[split]["density"], d[split]["detection"]
            best, bt = d[split]["detection_best"], d[split]["detection_best_threshold"]
            P(f"{name+' ['+setting+']':<44}{split:<10}"
              f"{den['mae']:7.2f}/{den['rmse']:<8.2f}"
              f"{det['mae']:7.2f}/{det['rmse']:<8.2f}"
              f"{best['mae']:8.2f}/{best['rmse']:<7.2f}({bt:.2f})")

    # old vs current test labels: re-score the dumped test predictions against
    # `original_pos_count`, the label version the CountEx paper reported on.
    P("\n" + "=" * 100)
    P("OLD vs CURRENT CoCount TEST LABELS  (150/2668 = 5.6% of test counts were revised)")
    P("=" * 100)
    try:
        from datasets import load_dataset, concatenate_datasets
        cats = ["FOO", "FUN", "OFF", "OTR", "HOU"]
        ds = load_dataset("BBVisual/CoCount-test")
        cache = {}
        for tag, d in res.items():
            f = os.path.join(MATCHED, f"{tag}_test_preds.json")
            if not os.path.exists(f):
                continue
            split = d.get("data_split", "ALL").upper()
            key = split
            if key not in cache:
                sub = concatenate_datasets([ds[c] for c in cats]) if key in ("ALL", "KC") else ds[key]
                cache[key] = (np.array(sub["pos_count"], float),
                              np.array(sub["original_pos_count"], float))
            cur, old_lbl = cache[key]
            recs = json.load(open(f))
            if len(recs) != len(cur):
                continue
            det = np.array([r["det"] for r in recs], float)
            den = np.array([r["den"] for r in recs], float)
            name, setting = NICE.get(tag, (tag, "?"))
            def mae(p, g):
                return float(np.abs(p - g).mean())
            P(f"{name+' ['+setting+']':<44} density {mae(den, old_lbl):6.2f}(old) -> {mae(den, cur):6.2f}(cur)   "
              f"det@0.42 {mae(det, old_lbl):6.2f}(old) -> {mae(det, cur):6.2f}(cur)")
    except Exception as e:
        P(f"  [old-label comparison unavailable: {e}]")

    # corrected-label delta
    P("\n" + "=" * 100)
    P("CORRECTED CoCount TEST LABELS (BBVisual/CoCount-test-corrected0621)")
    P("=" * 100)
    for tag, d in res.items():
        if "test" in d and "test_corr" in d:
            name, setting = NICE.get(tag, (tag, "?"))
            n = d["test_corr"].get("n_labels_changed_vs_original_test")
            P(f"{name+' ['+setting+']':<44} density {d['test']['density']['mae']:6.2f} -> {d['test_corr']['density']['mae']:6.2f}   "
              f"det@0.42 {d['test']['detection']['mae']:6.2f} -> {d['test_corr']['detection']['mae']:6.2f}   "
              f"({n} labels changed)")

    # seeds
    seeds = [t for t in ["full_seed666", "full_seed777", "full_seed888"] if t in res]
    if len(seeds) >= 2:
        P("\n" + "=" * 100)
        P("SEED VARIABILITY (density inference)")
        P("=" * 100)
        for split in ["val", "test"]:
            v = [res[t][split]["density"]["mae"] for t in seeds if split in res[t]]
            r = [res[t][split]["density"]["rmse"] for t in seeds if split in res[t]]
            if v:
                P(f"  {split}: MAE {np.mean(v):.2f} +/- {np.std(v):.2f}   RMSE {np.mean(r):.2f} +/- {np.std(r):.2f}   (n={len(v)} seeds)")

    # ---- NC aggregate over the five held-out supercategories ----
    nc_map = {"food": "FOO", "home": "HOU", "desk": "OFF", "misc": "OTR", "game": "FUN"}
    P("\n" + "=" * 100)
    P("NC AGGREGATE (mean over the five held-out supercategories, weighted by split size)")
    P("=" * 100)
    for method, pref in [("CountEx", "countex_nc_"), ("Ours", "ours_nc_"),
                         ("CountEx+MT", "mt_nc_"), ("CountEx+MT*", "mtstar_nc_"),
                         ("CountEx+OT-M", "otm_nc_")]:
        for split in ["val", "test"]:
            tags = [f"{pref}{d}" for d in nc_map if f"{pref}{d}" in res and split in res[f"{pref}{d}"]]
            if len(tags) < 5:
                continue
            def wmean(head, key):
                num = sum(res[t][split][head][key] * res[t][split][head]["n"] for t in tags)
                return num / sum(res[t][split][head]["n"] for t in tags)
            def wrmse(head):
                num = sum(res[t][split][head]["rmse"] ** 2 * res[t][split][head]["n"] for t in tags)
                return (num / sum(res[t][split][head]["n"] for t in tags)) ** 0.5
            P(f"  {method:<14} {split:<6} density {wmean('density','mae'):6.2f}/{wrmse('density'):6.2f}"
              f"   det@0.42 {wmean('detection','mae'):6.2f}/{wrmse('detection'):6.2f}")
        P("")

    # ---- localization ----
    loc_files = sorted(glob.glob("localization_eval/*.json"))
    if loc_files:
        P("=" * 100)
        P("LOCALIZATION on the held-out (never trained on) dot-labelled NC train splits")
        P("=" * 100)
        P(f"{'run':<26}{'imgs':>6}{'GAME0':>9}{'GAME1':>9}{'GAME2':>9}{'GAME3':>9}"
          f"{'detGAME3':>10}{'F1@.05':>9}{'F1@.10':>9}")
        for f in loc_files:
            d = json.load(open(f))
            g, gd = d["GAME_density"], d["GAME_detection"]
            pm = d["point_matching"]
            P(f"{os.path.basename(f)[:-5]:<26}{d['n_images']:>6}"
              f"{g['GAME0']:>9.2f}{g['GAME1']:>9.2f}{g['GAME2']:>9.2f}{g['GAME3']:>9.2f}"
              f"{gd['GAME3']:>10.2f}{pm['tol_0.05']['f1']:>9.3f}{pm['tol_0.1']['f1']:>9.3f}")

    # ---- PairTally ----
    pt_files = sorted(glob.glob("pairtally_eval/*.json"))
    pt_files = [f for f in pt_files if not f.endswith("_preds.json")]
    if pt_files:
        P("\n" + "=" * 100)
        P("PAIRTALLY (density inference unless noted)")
        P("=" * 100)
        for f in pt_files:
            d = json.load(open(f))
            den = d["density"]
            P(f"\n  {d['tag']}  (det@0.42 overall MAE {d['detection']['overall']['mae']:.2f}, "
              f"best thr {d['detection_best_threshold']:.2f} -> {d['detection_best_overall']['mae']:.2f})")
            for k in ["overall", "inter", "intra"]:
                if den.get(k):
                    P(f"    {k:<8} MAE {den[k]['mae']:6.2f}  RMSE {den[k]['rmse']:6.2f}  NAE {den[k]['nae']:.3f}")
            if den.get("intra_worst5pct"):
                w = den["intra_worst5pct"]
                P(f"    intra worst 5% (n={w['n']}): MAE {w['mae_of_worst']:.1f}, "
                  f"{100*w['share_of_intra_total_error']:.0f}% of all intra error")
                if den.get("intra_excl_worst5pct"):
                    e = den["intra_excl_worst5pct"]
                    P(f"    intra excluding them: MAE {e['mae']:6.2f}  RMSE {e['rmse']:6.2f}")
            for k in sorted(den):
                if k.startswith("INTRA_count_"):
                    P(f"    {k:<22} MAE {den[k]['mae']:6.2f} (n={den[k]['n']})")
            if "localization" in d:
                L = d["localization"]
                P(f"    GAME(density) 0-3: " + " ".join(f"{L['GAME_density'][f'GAME{i}']:.2f}" for i in range(4)))
                P(f"    point F1 @0.05 / @0.10: {L['point_matching']['tol_0.05']['f1']:.3f} / "
                  f"{L['point_matching']['tol_0.1']['f1']:.3f}")

    # ---- latency ----
    if os.path.exists("matched_eval/latency.json"):
        lat = json.load(open("matched_eval/latency.json"))
        P("\n" + "=" * 100)
        P(f"CLEAN LATENCY BENCHMARK ({lat.get('gpu','?')}, bf16, batch 1, same images back to back)")
        P("=" * 100)
        for k in ["countex", "ours", "ours_no_prior"]:
            if k in lat:
                P(f"  {k:<16} {lat[k]['median_ms']:7.1f} ms median   {lat[k]['mean_ms']:7.1f} ms mean")

    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
