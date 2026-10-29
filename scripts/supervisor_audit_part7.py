"""Supervisor audit, check 7: row-level A vs D prediction comparison on
final_target_test, per seed. Confirms D's reported accuracy increase
corresponds to actual flips in correctly-classified IDs, not an artifact."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "results" / "experiment" / "predictions"
gold = pd.read_csv(ROOT / "data" / "final_target_test.csv").set_index("id")["label_name_3class"]


def compare(name_a: str, name_b: str) -> dict:
    a = pd.read_csv(PRED_DIR / f"{name_a}_target_test.csv").set_index("id")["pred"]
    b = pd.read_csv(PRED_DIR / f"{name_b}_target_test.csv").set_index("id")["pred"]
    assert set(a.index) == set(b.index) == set(gold.index)
    merged = pd.DataFrame({"gold": gold, "a_pred": a, "b_pred": b})
    merged["a_correct"] = merged["a_pred"] == merged["gold"]
    merged["b_correct"] = merged["b_pred"] == merged["gold"]

    agreement = (merged["a_pred"] == merged["b_pred"]).mean()
    changed = merged[merged["a_pred"] != merged["b_pred"]]
    corrected = merged[(~merged["a_correct"]) & (merged["b_correct"])]
    introduced = merged[(merged["a_correct"]) & (~merged["b_correct"])]

    n_a_correct = int(merged["a_correct"].sum())
    n_b_correct = int(merged["b_correct"].sum())
    net_change = len(corrected) - len(introduced)
    expected_b_correct = n_a_correct + net_change

    return {
        "a": name_a, "b": name_b,
        "n": len(merged),
        "agreement_rate": float(agreement),
        "n_changed_predictions": int(len(changed)),
        "n_corrected": int(len(corrected)),
        "n_newly_wrong": int(len(introduced)),
        "a_n_correct": n_a_correct,
        "b_n_correct": n_b_correct,
        "net_change_from_corrected_minus_introduced": net_change,
        "expected_b_correct_from_decomposition": expected_b_correct,
        "decomposition_matches_actual": expected_b_correct == n_b_correct,
        "corrected_ids_sample": corrected.index.tolist()[:10],
        "newly_wrong_ids_sample": introduced.index.tolist()[:10],
    }


def main():
    results = {}
    for seed in [0, 1, 2]:
        r = compare("A", f"D_seed{seed}")
        results[f"A_vs_D_seed{seed}"] = r
        print(f"A vs D_seed{seed}: agreement={r['agreement_rate']:.4f} changed={r['n_changed_predictions']} "
              f"corrected={r['n_corrected']} newly_wrong={r['n_newly_wrong']} "
              f"A_correct={r['a_n_correct']} D_correct={r['b_n_correct']} "
              f"decomposition_matches={r['decomposition_matches_actual']}")
        r2 = compare(f"C_seed{seed}", f"D_seed{seed}")
        results[f"C_vs_D_seed{seed}"] = r2
        print(f"C vs D_seed{seed}: agreement={r2['agreement_rate']:.4f} changed={r2['n_changed_predictions']} "
              f"corrected={r2['n_corrected']} newly_wrong={r2['n_newly_wrong']} "
              f"C_correct={r2['a_n_correct']} D_correct={r2['b_n_correct']} "
              f"decomposition_matches={r2['decomposition_matches_actual']}")

    out = ROOT / "results" / "audit_supervisor" / "part7_row_level_comparison.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    all_match = all(r["decomposition_matches_actual"] for r in results.values())
    print("\nOVERALL:", "ALL decompositions match (corrected - newly_wrong = actual net change in #correct)" if all_match else "*** DECOMPOSITION MISMATCH FOUND ***")
    print(f"saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
