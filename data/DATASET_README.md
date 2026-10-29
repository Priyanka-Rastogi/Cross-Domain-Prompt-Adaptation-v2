# data/ -- provenance, schema, and known limitations

Built by `src/prepare_data.py`. Full provenance (exact HF revisions, row
counts, label distributions, SHA-256 hashes, duplicate/overlap checks) is
re-generated on every run into `artifacts/manifests/data_manifest.json` --
that file is the authoritative, reproducible record; this document explains
it in prose. See `docs/DATA_VERIFICATION.md` for the full dataset
verification writeup (schemas, threshold verification, licenses, citations).

## Files and roles

| File | Role | N | Source | Official upstream split |
|---|---|---:|---|---|
| `source_train.csv` | instruction-proposal / demonstration pool | 8544 | SST | train (full) |
| `source_validation.csv` | prompt/demonstration selection (source labels only) | 1101 | SST | validation (full) |
| `source_test.csv` | final in-domain evaluation | 2210 | SST | test (full) |
| `target_dev.csv` | screening / model-setting selection ONLY | 500 | Yelp Review Full | train (stratified sample, seed 42) |
| `final_target_test.csv` | final cross-domain evaluation (untouched until all settings frozen) | 1000 | Yelp Review Full | test (stratified sample, seed 42) |

SST's official `train`/`validation`/`test` splits are used in full (no further
re-splitting by this project) -- 8544/1101/2210 is the standard SST
sentence-level split size, so no extra justification was needed for sample
sizes there. Yelp's two official splits (650,000 / 50,000 rows, exactly
balanced across 5 star-classes -- verified, not assumed) are far too large to
use whole for repeated small-model inference during iterative prompt search;
`target_dev` (500, stratified ~100/class) and `final_target_test` (1000,
stratified ~200/class) were chosen as affordable sample sizes for Qwen2.5
0.5B/1.5B-Instruct scoring on this GPU (each full screening/evaluation pass
is a few hundred forward passes per label option, well under a minute of GPU
time at the measured ~33GB free VRAM -- see `docs/ENVIRONMENT.md`).

## Required schema

`id,text,label,label_name,domain,original_split`, plus two extra columns
carried on every row: `label_3class,label_name_3class` (the three_class view
of the same row -- see `docs/DATA_VERIFICATION.md` for the collapse rule).
`label`/`label_name` are the five_class view by default. Text is stored in
full, untruncated form; any inference-time truncation is applied later, at
prompt-build time, per the fixed policy in `configs/protocol.yaml`.

## Stable IDs

`sst_<official_split>_<row_index>` for source rows (0-indexed row order
within the pinned parquet commit); `yelp_<official_split>_<original_row_index>`
for target rows (original row index within Yelp's own train/test split, so a
given Yelp review's ID is stable and traceable even though only 500/1000 of
the 650k/50k rows were sampled). No duplicate IDs exist within any file
(verified programmatically; see manifest `duplicate_and_overlap_checks`).

## Known limitations (disclosed, not silently fixed)

- **No original Stanford sentence IDs**: the HF parquet mirror used for SST
  does not carry Stanford's original `sentence_index`/phrase IDs from the raw
  archive (unreachable from this environment; see `docs/DATA_VERIFICATION.md`).
  IDs here are this project's own, stable within this project only.
- **No Yelp review/business/user IDs**: the Zhang-et-al. processed release
  strips all such metadata, so group-aware disjointness (e.g. by business or
  reviewer) cannot be enforced or verified -- only split-level (official
  train vs. test) and text-level disjointness checks were possible.
- **A few exact-duplicate / cross-split-duplicate sentences exist in the
  official SST split itself** (not introduced by this project): 10 duplicate
  normalized-text pairs within `source_train` (e.g. "I like it .",
  "One of the worst movies of the year ." each appear twice, both times with
  the same label), 1 within `source_validation`, and 3 short generic phrases
  ("hey arnold !", "what 's next ?", "no. .") that appear in both
  `source_train` and one of `source_validation`/`source_test`. These were
  inspected manually: all are either short generic phrases or exact repeated
  fragments with **consistent labels** across their occurrences, not
  label-inconsistent noise. Given this affects at most 3 rows out of 3311
  source dev+test rows (0.09%), and they originate from the official,
  unmodified upstream split, no rows were removed -- removing official split
  members ourselves would itself be an undisclosed change to a standard
  benchmark split. This is disclosed here rather than silently present only
  in the raw manifest.
- `target_dev` and `final_target_test` were checked for cross-file normalized
  text overlap: **zero overlapping rows found** (consistent with Yelp's
  train/test splits being genuinely disjoint reviews, at least among the
  sampled 500 + 1000 rows).

## Licenses (recorded accurately, not assumed)

- SST: `license: unknown` per the HF dataset's own card; Stanford's original
  release page states no formal license either. Treated as an unclear-license
  research benchmark.
- Yelp Review Full: governed by the official Yelp Dataset Agreement (see
  `docs/DATA_VERIFICATION.md` for the link) -- a real, non-public-domain
  usage license, carried over accurately from the prior project's handling of
  Yelp Polarity under the same agreement.
