"""Section 7 data preparation: build the frozen, separated CSV datasets.

Produces, under data/:
  source_train.csv        full SST 'train' split (8544 rows) -- instruction
                           proposal / demonstration pool source
  source_validation.csv   full SST 'validation' split (1101 rows) -- prompt
                           selection (source-domain labels only)
  source_test.csv         full SST 'test' split (2210 rows) -- final in-domain eval
  target_dev.csv          stratified sample of Yelp 'train' split (500 rows)
                           -- screening / model-setting selection ONLY
  final_target_test.csv   stratified sample of Yelp 'test' split (1000 rows)
                           -- final cross-domain eval, untouched until freeze

Every row carries both the five_class and three_class label (three_class is a
deterministic function of five_class; see docs/DATA_VERIFICATION.md), plus the
required columns `id,text,label,label_name,domain,original_split` where
`label`/`label_name` default to the five_class view
(`label_3class`/`label_name_3class` carry the three_class view).

Run with the project venv: .venv/bin/python src/prepare_data.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST_DIR = ROOT / "artifacts" / "manifests"

SST_REPO = "stanfordnlp/sst"
SST_REVISION = "8a7d55ee01d70b8ae53e0b3c66a83b2a6198c2a7"  # refs/convert/parquet
YELP_REPO = "Yelp/yelp_review_full"
YELP_REVISION = "c1f9ee939b7d05667af864ee1cb066393154bf85"

SEED = 42
TARGET_DEV_N = 500
FINAL_TARGET_TEST_N = 1000

FIVE_CLASS_NAMES = ["very_negative", "negative", "neutral", "positive", "very_positive"]
THREE_CLASS_NAMES = ["negative", "neutral", "positive"]
FIVE_TO_THREE = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}  # very_neg/neg->neg, neutral->neutral, pos/very_pos->pos


def bucket_sst_score(score: float) -> int:
    """Verified in scripts/verify_sst_thresholds.py (100% agreement, n=8220)."""
    if score <= 0.2:
        return 0
    if score <= 0.4:
        return 1
    if score <= 0.6:
        return 2
    if score <= 0.8:
        return 3
    return 4


def load_sst_split(split_file: str) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=SST_REPO,
        repo_type="dataset",
        revision=SST_REVISION,
        filename=f"default/{split_file}/0000.parquet",
    )
    df = pd.read_parquet(path)[["sentence", "label"]].reset_index(drop=True)
    df["label_5"] = df["label"].apply(bucket_sst_score)
    df["label_3"] = df["label_5"].map(FIVE_TO_THREE)
    df = df.rename(columns={"sentence": "text"})
    return df


def build_source_csv(split_file: str, original_split: str, id_prefix: str) -> pd.DataFrame:
    df = load_sst_split(split_file)
    out = pd.DataFrame(
        {
            "id": [f"{id_prefix}_{i}" for i in range(len(df))],
            "text": df["text"],
            "label": df["label_5"],
            "label_name": df["label_5"].map(lambda i: FIVE_CLASS_NAMES[i]),
            "label_3class": df["label_3"],
            "label_name_3class": df["label_3"].map(lambda i: THREE_CLASS_NAMES[i]),
            "domain": "movie",
            "original_split": original_split,
        }
    )
    return out


def stratified_sample(df: pd.DataFrame, n: int, strat_col: str, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n_classes = df[strat_col].nunique()
    per_class = n // n_classes
    remainder = n - per_class * n_classes
    parts = []
    class_values = sorted(df[strat_col].unique())
    for i, cls in enumerate(class_values):
        take = per_class + (1 if i < remainder else 0)
        pool = df[df[strat_col] == cls]
        idx = rng.choice(pool.index.values, size=take, replace=False)
        parts.append(df.loc[idx])
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def build_target_csv(official_split: str, n: int, id_prefix: str, seed: int) -> pd.DataFrame:
    ds = load_dataset(YELP_REPO, revision=YELP_REVISION)[official_split]
    df = ds.to_pandas().reset_index(drop=True)
    df["orig_row"] = df.index
    sampled = stratified_sample(df, n, "label", seed)
    out = pd.DataFrame(
        {
            "id": [f"{id_prefix}_{r}" for r in sampled["orig_row"]],
            "text": sampled["text"],
            "label": sampled["label"],
            "label_name": sampled["label"].map(lambda i: FIVE_CLASS_NAMES[i]),
            "label_3class": sampled["label"].map(FIVE_TO_THREE),
            "label_name_3class": sampled["label"].map(lambda i: THREE_CLASS_NAMES[FIVE_TO_THREE[i]]),
            "domain": "yelp",
            "original_split": official_split,
        }
    )
    return out.reset_index(drop=True)


def normalize_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def check_duplicates_and_overlap(frames: dict[str, pd.DataFrame]) -> dict:
    report: dict = {"within_role_duplicate_ids": {}, "within_role_duplicate_text": {}, "cross_role_text_overlap": {}}
    norm_text_by_role = {}
    for name, df in frames.items():
        dup_ids = df["id"][df["id"].duplicated()].tolist()
        report["within_role_duplicate_ids"][name] = dup_ids
        norm = df["text"].apply(normalize_text)
        norm_text_by_role[name] = set(norm)
        dup_text = norm[norm.duplicated()].tolist()
        report["within_role_duplicate_text"][name] = len(dup_text)
    names = list(frames.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = norm_text_by_role[a] & norm_text_by_role[b]
            if overlap:
                report["cross_role_text_overlap"][f"{a}__{b}"] = len(overlap)
    return report


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    source_train = build_source_csv("train", "train", "sst_train")
    source_validation = build_source_csv("validation", "validation", "sst_validation")
    source_test = build_source_csv("test", "test", "sst_test")

    target_dev = build_target_csv("train", TARGET_DEV_N, "yelp_train", SEED)
    final_target_test = build_target_csv("test", FINAL_TARGET_TEST_N, "yelp_test", SEED)

    frames = {
        "source_train": source_train,
        "source_validation": source_validation,
        "source_test": source_test,
        "target_dev": target_dev,
        "final_target_test": final_target_test,
    }

    for name, df in frames.items():
        out_path = DATA_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)

    dup_report = check_duplicates_and_overlap(frames)

    manifest = {
        "seed": SEED,
        "source_dataset": {
            "hf_repo": SST_REPO,
            "hf_revision": SST_REVISION,
            "config": "default",
        },
        "target_dataset": {
            "hf_repo": YELP_REPO,
            "hf_revision": YELP_REVISION,
            "config": "yelp_review_full",
        },
        "row_counts": {name: len(df) for name, df in frames.items()},
        "label_distribution_5class": {
            name: df["label_name"].value_counts().to_dict() for name, df in frames.items()
        },
        "label_distribution_3class": {
            name: df["label_name_3class"].value_counts().to_dict() for name, df in frames.items()
        },
        "duplicate_and_overlap_checks": dup_report,
        "file_sha256": {
            name: sha256_of_file(DATA_DIR / f"{name}.csv") for name in frames
        },
    }
    (MANIFEST_DIR / "data_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
