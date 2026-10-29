"""Supervisor audit, check 8: normalized-text overlap across ALL experimental
roles (not just the pairs checked in Section 7's data_manifest.json), plus a
direct check that no demonstration actually used in C/D is textually
identical to the query it was shown alongside (which would leak the answer)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def main():
    files = {
        "source_train": "data/source_train.csv",
        "source_validation": "data/source_validation.csv",
        "source_test": "data/source_test.csv",
        "target_dev": "data/target_dev.csv",
        "final_target_test": "data/final_target_test.csv",
    }
    norm_text = {}
    for name, path in files.items():
        df = pd.read_csv(ROOT / path)
        norm_text[name] = set(df["text"].apply(norm))

    print("=== pairwise normalized-text overlap across ALL role pairs ===")
    names = list(files.keys())
    overlap_report = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ov = norm_text[a] & norm_text[b]
            overlap_report[f"{a}__{b}"] = len(ov)
            print(f"  {a:18s} x {b:18s} : {len(ov)} overlapping normalized texts")

    print()
    print("=== group IDs available? ===")
    print("  SST (source): no original Stanford sentence_index/phrase IDs in the HF mirror used (documented in docs/DATA_VERIFICATION.md).")
    print("  Yelp (target): no business/user/review IDs in the Zhang-et-al. processed release (documented in data/DATASET_README.md).")
    print("  -> group-aware separation beyond split-level + text-level is not possible with these releases; this is a disclosed, not silently ignored, limitation.")

    print()
    print("=== demonstration-vs-query leakage check: does any demo text used in C/D equal the query it accompanied? ===")
    leaks = []
    for seed in [0, 1, 2]:
        c_meta = json.loads((ROOT / "results" / "experiment" / "metrics" / f"C_seed{seed}_final.json").read_text())
        d_meta = json.loads((ROOT / "results" / "experiment" / "metrics" / f"D_seed{seed}_final.json").read_text())
        for cond_name, meta in [(f"C_seed{seed}", c_meta), (f"D_seed{seed}", d_meta)]:
            demo_texts = {norm(t) for t, _lab in meta["demos"]}
            # demos must come only from source_train
            not_in_source_train = demo_texts - norm_text["source_train"]
            if not_in_source_train:
                leaks.append((cond_name, "demo_not_in_source_train", list(not_in_source_train)[:3]))
            # demos must never equal any source_test or final_target_test query text (the only two
            # eval sets this condition's demos were paired with)
            overlap_with_source_test = demo_texts & norm_text["source_test"]
            overlap_with_target_test = demo_texts & norm_text["final_target_test"]
            if overlap_with_source_test:
                leaks.append((cond_name, "demo_overlaps_source_test_query", list(overlap_with_source_test)))
            if overlap_with_target_test:
                leaks.append((cond_name, "demo_overlaps_target_test_query", list(overlap_with_target_test)))
            print(f"  {cond_name}: {len(meta['demos'])} demos, all from source_train={not bool(not_in_source_train)}, "
                  f"overlap_with_source_test={len(overlap_with_source_test)}, overlap_with_target_test={len(overlap_with_target_test)}")

    out = ROOT / "results" / "audit_supervisor" / "part8_overlap_and_leakage.json"
    out.write_text(json.dumps({"pairwise_overlap": overlap_report, "leakage_issues": leaks}, indent=2, default=str))
    print()
    print("OVERALL:", "*** LEAKAGE OR SOURCING ISSUES FOUND ***" if leaks else "ALL CLEAN -- every demo used in C/D comes from source_train only, with zero overlap with any evaluated query text.")
    print(f"saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
