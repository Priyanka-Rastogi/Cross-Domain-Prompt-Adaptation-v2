"""Supervisor audit, check 2: re-derive gold labels from the ORIGINAL data
CSVs (source_test.csv / final_target_test.csv) by ID join -- do not trust the
'gold' column already embedded in results/experiment/predictions/*.csv, since
that embedding is itself part of what is being audited. Report exact
correct/total counts per condition/seed, and flag any missing/duplicate/extra
row or any mismatch between the freshly-joined gold and the embedded 'gold'
column (which would indicate a bug in how predictions were written)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "results" / "experiment" / "predictions"
LABELS = ["negative", "neutral", "positive"]

source_gold = pd.read_csv(ROOT / "data" / "source_test.csv").set_index("id")["label_name_3class"]
target_gold = pd.read_csv(ROOT / "data" / "final_target_test.csv").set_index("id")["label_name_3class"]


def audit_one(name: str, split: str) -> dict:
    gold_series = source_gold if split == "source_test" else target_gold
    pred_path = PRED_DIR / f"{name}_{split}.csv"
    pred = pd.read_csv(pred_path)

    report = {"file": str(pred_path.relative_to(ROOT))}
    report["n_rows"] = len(pred)
    report["duplicate_ids"] = int(pred["id"].duplicated().sum())

    gold_ids = set(gold_series.index)
    pred_ids = set(pred["id"])
    report["missing_ids"] = len(gold_ids - pred_ids)
    report["extra_ids"] = len(pred_ids - gold_ids)

    # independent join by id -- do NOT rely on row order
    merged = pred.merge(gold_series.rename("gold_from_source_file"), left_on="id", right_index=True, how="inner")
    report["n_joined"] = len(merged)
    assert len(merged) == len(gold_series), f"{name}/{split}: join did not cover every gold row"

    # cross-check: does the embedded 'gold' column match the freshly re-joined gold?
    mismatches = merged[merged["gold"] != merged["gold_from_source_file"]]
    report["embedded_gold_vs_rejoined_gold_mismatches"] = len(mismatches)
    if len(mismatches) > 0:
        report["mismatch_sample_ids"] = mismatches["id"].head(5).tolist()

    y_true = merged["gold_from_source_file"].tolist()
    y_pred = merged["pred"].tolist()
    n_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    report["correct"] = n_correct
    report["total"] = len(y_true)
    report["accuracy"] = accuracy_score(y_true, y_pred)
    report["accuracy_fraction"] = f"{n_correct}/{len(y_true)}"
    report["macro_f1"] = f1_score(y_true, y_pred, labels=LABELS, average="macro")
    report["predicted_class_counts"] = merged["pred"].value_counts().to_dict()
    return report


def main():
    names = ["A"] + [f"{c}_seed{s}" for c in ["B", "C", "D"] for s in [0, 1, 2]]
    all_reports = {}
    for name in names:
        for split in ["source_test", "target_test"]:
            key = f"{name}_{split}"
            r = audit_one(name, split)
            all_reports[key] = r
            flag = "OK" if r["missing_ids"] == 0 and r["extra_ids"] == 0 and r["duplicate_ids"] == 0 and r["embedded_gold_vs_rejoined_gold_mismatches"] == 0 else "*** ISSUE ***"
            print(f"{key:28s} n={r['total']:4d} correct={r['correct']:4d} acc={r['accuracy']:.4f} "
                  f"macro_f1={r['macro_f1']:.4f} missing={r['missing_ids']} extra={r['extra_ids']} "
                  f"dup={r['duplicate_ids']} embedded_gold_mismatch={r['embedded_gold_vs_rejoined_gold_mismatches']}  {flag}")

    out = ROOT / "results" / "audit_supervisor" / "part2_rejoined_metrics.json"
    out.write_text(json.dumps(all_reports, indent=2, default=str))
    print(f"\nsaved: {out.relative_to(ROOT)}")

    any_issue = any(
        r["missing_ids"] or r["extra_ids"] or r["duplicate_ids"] or r["embedded_gold_vs_rejoined_gold_mismatches"]
        for r in all_reports.values()
    )
    print("\nOVERALL:", "*** AT LEAST ONE ISSUE FOUND ***" if any_issue else "ALL CLEAN -- independently-rejoined gold matches embedded gold, no missing/extra/duplicate rows anywhere.")


if __name__ == "__main__":
    main()
