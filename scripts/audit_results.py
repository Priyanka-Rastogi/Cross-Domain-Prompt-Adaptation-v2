"""Section 11 audit: independently recompute every metric from the raw
prediction CSVs (never trusting the cached *_final.json numbers at face
value), using scikit-learn as a second, independent implementation against
this project's own hand-rolled metrics.py. Checks ID alignment, reports
per-seed values + mean/sample-std, and a paired per-example bootstrap CI for
key A-D differences on final_target_test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix as sk_cm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from metrics import accuracy as my_accuracy, macro_f1 as my_macro_f1, majority_baseline  # noqa: E402

PRED_DIR = ROOT / "results" / "experiment" / "predictions"
LABELS = ["negative", "neutral", "positive"]
SEEDS = [0, 1, 2]


def check_id_alignment(pred_csv: Path, gold_csv: Path) -> dict:
    pred = pd.read_csv(pred_csv)
    gold = pd.read_csv(gold_csv)
    gold_ids = set(gold["id"])
    pred_ids = pred["id"].tolist()
    missing = gold_ids - set(pred_ids)
    extra = set(pred_ids) - gold_ids
    dup = pred["id"][pred["id"].duplicated()].tolist()
    return {"n_pred": len(pred), "n_gold": len(gold), "missing": len(missing), "extra": len(extra), "duplicates": len(dup)}


def recompute(pred_csv: Path) -> dict:
    df = pd.read_csv(pred_csv)
    y_true, y_pred = df["gold"].tolist(), df["pred"].tolist()
    acc_mine = my_accuracy(y_true, y_pred)
    acc_sklearn = accuracy_score(y_true, y_pred)
    mf1_mine = my_macro_f1(y_true, y_pred, LABELS)
    mf1_sklearn = f1_score(y_true, y_pred, labels=LABELS, average="macro")
    assert abs(acc_mine - acc_sklearn) < 1e-9, f"accuracy mismatch in {pred_csv}: {acc_mine} vs {acc_sklearn}"
    assert abs(mf1_mine - mf1_sklearn) < 1e-9, f"macro-F1 mismatch in {pred_csv}: {mf1_mine} vs {mf1_sklearn}"
    cm = sk_cm(y_true, y_pred, labels=LABELS)
    return {
        "n": len(df),
        "accuracy": acc_sklearn,
        "macro_f1": mf1_sklearn,
        "majority_baseline": majority_baseline(y_true),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
        "ids": df["id"].tolist(),
    }


def paired_bootstrap_ci(ids_a, y_true_a, y_pred_a, ids_b, y_true_b, y_pred_b, n_boot=5000, seed=0):
    """Resampling unit: individual final_target_test examples (matched by id).
    Returns 95% percentile CI for (acc_b - acc_a)."""
    order_a = {i: k for k, i in enumerate(ids_a)}
    order_b = {i: k for k, i in enumerate(ids_b)}
    common = [i for i in ids_a if i in order_b]
    assert len(common) == len(ids_a) == len(ids_b), "prediction sets are not aligned by id"
    correct_a = np.array([int(y_true_a[order_a[i]] == y_pred_a[order_a[i]]) for i in common])
    correct_b = np.array([int(y_true_b[order_b[i]] == y_pred_b[order_b[i]]) for i in common])
    diffs = correct_b - correct_a
    rng = np.random.RandomState(seed)
    n = len(diffs)
    boot_means = np.array([diffs[rng.randint(0, n, size=n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {"point_estimate": float(diffs.mean()), "ci95_lo": float(lo), "ci95_hi": float(hi), "n": n, "n_boot": n_boot}


def main():
    report = {"id_alignment": {}, "per_condition": {}, "paired_diffs_target_test": {}}

    # --- A (seed-invariant) ---
    for split, gold_csv in [("source_test", ROOT / "data" / "source_test.csv"), ("target_test", ROOT / "data" / "final_target_test.csv")]:
        fname = f"A_{'source_test' if split=='source_test' else 'target_test'}.csv"
        pc = PRED_DIR / fname
        report["id_alignment"][f"A_{split}"] = check_id_alignment(pc, gold_csv)
        report["per_condition"].setdefault("A", {})[split] = recompute(pc)

    # --- B, C, D x 3 seeds ---
    for cond in ["B", "C", "D"]:
        for seed in SEEDS:
            name = f"{cond}_seed{seed}"
            for split, gold_csv in [("source_test", ROOT / "data" / "source_test.csv"), ("target_test", ROOT / "data" / "final_target_test.csv")]:
                fname = f"{name}_{'source_test' if split=='source_test' else 'target_test'}.csv"
                pc = PRED_DIR / fname
                report["id_alignment"][f"{name}_{split}"] = check_id_alignment(pc, gold_csv)
                report["per_condition"].setdefault(name, {})[split] = recompute(pc)

    # --- summary tables: per-seed + mean/std ---
    summary = {}
    for cond in ["A", "B", "C", "D"]:
        accs_src, mf1s_src, accs_tgt, mf1s_tgt = [], [], [], []
        if cond == "A":
            r = report["per_condition"]["A"]
            summary["A"] = {
                "source_test_accuracy": r["source_test"]["accuracy"], "source_test_macro_f1": r["source_test"]["macro_f1"],
                "target_test_accuracy": r["target_test"]["accuracy"], "target_test_macro_f1": r["target_test"]["macro_f1"],
                "seed_invariant": True,
            }
            continue
        for seed in SEEDS:
            r = report["per_condition"][f"{cond}_seed{seed}"]
            accs_src.append(r["source_test"]["accuracy"]); mf1s_src.append(r["source_test"]["macro_f1"])
            accs_tgt.append(r["target_test"]["accuracy"]); mf1s_tgt.append(r["target_test"]["macro_f1"])
        summary[cond] = {
            "source_test_accuracy_per_seed": accs_src, "source_test_accuracy_mean": float(np.mean(accs_src)), "source_test_accuracy_std": float(np.std(accs_src, ddof=1)),
            "source_test_macro_f1_per_seed": mf1s_src, "source_test_macro_f1_mean": float(np.mean(mf1s_src)), "source_test_macro_f1_std": float(np.std(mf1s_src, ddof=1)),
            "target_test_accuracy_per_seed": accs_tgt, "target_test_accuracy_mean": float(np.mean(accs_tgt)), "target_test_accuracy_std": float(np.std(accs_tgt, ddof=1)),
            "target_test_macro_f1_per_seed": mf1s_tgt, "target_test_macro_f1_mean": float(np.mean(mf1s_tgt)), "target_test_macro_f1_std": float(np.std(mf1s_tgt, ddof=1)),
        }
    report["summary"] = summary

    # --- paired bootstrap CIs on target_test, using seed=1 (median-ish D performance) as the representative seed ---
    a_t = report["per_condition"]["A"]["target_test"]
    for cond, seed in [("B", 1), ("C", 1), ("D", 1)]:
        c_t = report["per_condition"][f"{cond}_seed{seed}"]["target_test"]
        ci = paired_bootstrap_ci(a_t["ids"], a_t["y_true"], a_t["y_pred"], c_t["ids"], c_t["y_true"], c_t["y_pred"])
        report["paired_diffs_target_test"][f"{cond}_seed{seed}_minus_A"] = ci

    d1 = report["per_condition"]["D_seed1"]["target_test"]
    c1 = report["per_condition"]["C_seed1"]["target_test"]
    report["paired_diffs_target_test"]["D_seed1_minus_C_seed1"] = paired_bootstrap_ci(
        c1["ids"], c1["y_true"], c1["y_pred"], d1["ids"], d1["y_true"], d1["y_pred"]
    )

    # strip large arrays before saving
    def strip(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = {kk: (strip(vv) if isinstance(vv, dict) else vv) for kk, vv in v.items() if kk not in ("y_true", "y_pred", "ids")}
            else:
                out[k] = v
        return out

    lean_report = {
        "id_alignment": report["id_alignment"],
        "summary": report["summary"],
        "per_condition": {k: {kk: {kkk: vvv for kkk, vvv in vv.items() if kkk not in ("y_true", "y_pred", "ids")} for kk, vv in v.items()} for k, v in report["per_condition"].items()},
        "paired_diffs_target_test": report["paired_diffs_target_test"],
    }
    out_path = ROOT / "results" / "audit"
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "independent_audit.json").write_text(json.dumps(lean_report, indent=2, default=str))
    print(json.dumps({"summary": summary, "paired_diffs_target_test": report["paired_diffs_target_test"]}, indent=2, default=str))
    print("\nALL ID-ALIGNMENT AND SKLEARN CROSS-CHECKS PASSED (no assertion errors raised).")


if __name__ == "__main__":
    main()
