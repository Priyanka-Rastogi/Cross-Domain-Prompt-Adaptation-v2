"""One-time verification that the standard SST-5 bucket thresholds
(<=0.2/<=0.4/<=0.6/<=0.8/>0.8) match an independently-discretized mirror.

Not part of the data pipeline; kept for reproducibility of the check recorded
in docs/DATA_VERIFICATION.md. Run with the project venv:
    .venv/bin/python scripts/verify_sst_thresholds.py
"""

import json

import pandas as pd
from huggingface_hub import hf_hub_download

SST_PARQUET_REVISION = "8a7d55ee01d70b8ae53e0b3c66a83b2a6198c2a7"
SETFIT_SST5_REVISION = "e51bdcd8cd3a30da231967c1a249ba59361279a3"


def bucket(score: float) -> int:
    if score <= 0.2:
        return 0
    if score <= 0.4:
        return 1
    if score <= 0.6:
        return 2
    if score <= 0.8:
        return 3
    return 4


def normalize(text: str) -> str:
    return text.strip().lower().replace("``", '"').replace("''", '"')


def main() -> None:
    cont_path = hf_hub_download(
        repo_id="stanfordnlp/sst",
        repo_type="dataset",
        revision=SST_PARQUET_REVISION,
        filename="default/train/0000.parquet",
    )
    cont = pd.read_parquet(cont_path)[["sentence", "label"]].rename(
        columns={"label": "cont_label"}
    )

    disc_path = hf_hub_download(
        repo_id="SetFit/sst5",
        repo_type="dataset",
        revision=SETFIT_SST5_REVISION,
        filename="train.jsonl",
    )
    disc = pd.DataFrame(json.loads(line) for line in open(disc_path))

    cont["key"] = cont["sentence"].apply(normalize)
    disc["key"] = disc["text"].apply(normalize)
    merged = cont.merge(disc, on="key", how="inner")

    merged["bucket_pred"] = merged["cont_label"].apply(bucket)
    agree = (merged["bucket_pred"] == merged["label"]).mean()

    print(f"matched rows: {len(merged)} / {len(cont)} continuous, {len(disc)} discrete")
    print(f"threshold agreement rate: {agree:.6f}")
    assert agree == 1.0, "threshold scheme does not perfectly match the independent mirror"


if __name__ == "__main__":
    main()
