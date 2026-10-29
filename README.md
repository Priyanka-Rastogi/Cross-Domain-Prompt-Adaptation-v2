# Cross-Domain Prompt Adaptation v2

Frozen-weight, in-context prompt adaptation for cross-domain sentiment
classification: can a natural-language prompt (instruction + demonstrations),
learned from labeled **source-domain** examples, transfer to an **unseen
target domain**, using only small frozen pretrained LMs (no fine-tuning,
LoRA, adapters, soft prompts, or gradient updates)?

Source domain: Stanford Sentiment Treebank (movie-review sentences).
Target domain: Yelp Review Full (business reviews). Model: Qwen2.5-0.5B-Instruct,
weights frozen throughout (verified zero trainable parameters before any run).

This is a dissertation support project (M.Tech CSE, Graphic Era, supervised by
Dr. Narayan Chaturvedi). **Scope note:** jointly optimizing instructions and
demonstrations is not claimed as novel here (see Wan et al., "Teach Better or
Show Smarter?"; EASE, Wu et al.; MIPRO/MIPROv2) -- this repository is an
experimental foundation and reproducibility artifact, not a manuscript or a
claim of a new method.

This is an independent, from-scratch project. It is isolated from an earlier,
separate repository (`Qwen-Cross-Domain-Prompt-Adaptation`) and reuses none of
that project's predictions, selected prompts, or evaluation caches.

## What this is / isn't

- "Adaptation" = selecting a natural-language instruction and/or a small set
  of demonstrations by search over frozen-model scores. No model weights are
  ever updated.
- Classification is done by **complete-label conditional likelihood**
  (teacher-forced, length-normalized mean log-probability per answer token
  over the verbalizer options), not free-form generation.
- Four conditions are compared on the same frozen model: **A** fixed baseline
  instruction, **B** optimized instruction only, **C** fixed instruction +
  random demonstrations, **D** jointly optimized instruction + demonstrations.
  3 seeds each for B/C/D.

## Results summary

Full results, per-class breakdowns, confusion matrices, uncertainty
intervals, and cost accounting are in [`results/FINAL_REPORT.md`](results/FINAL_REPORT.md).
A supervisor-requested independent forensic audit of these results (prompted
by a suspected evaluation bug in the reported accuracy) is in
[`results/audit_supervisor/SUPERVISOR_AUDIT_REPORT.md`](results/audit_supervisor/SUPERVISOR_AUDIT_REPORT.md)
-- conclusion: no bug found; the accuracy/macro-F1 gap is explained by the
baseline's near-total failure to predict the "neutral" class, which D's
prompt search measurably and consistently improves.

**Cross-domain (target_test, Yelp, n=1000, balanced 400/200/400):**

| Condition | Accuracy (mean ± sd) | Macro-F1 (mean ± sd) |
|---|---|---|
| A | 0.770 | 0.580 |
| B | 0.769 ± 0.023 | 0.682 ± 0.046 |
| C | 0.775 ± 0.002 | 0.637 ± 0.003 |
| **D** | **0.791 ± 0.005** | **0.717 ± 0.018** |

## Repository layout

```
configs/      frozen experimental protocol (configs/protocol.yaml) -- written
              and committed before any model inference was run
docs/         environment setup rationale; dataset verification write-up
              (schemas, label meanings, license checks, threshold verification)
data/         dataset README (schema/provenance/known limitations).
              The CSVs themselves are NOT committed -- see "Dataset access" below.
src/          scoring interface, model loading, data prep, instruction
              proposal, demo/instruction search, experiment orchestration
scripts/      standalone verification/audit scripts (data threshold check,
              independent metric audit, supervisor forensic audit)
tests/        live-model tests of the scoring interface (token boundaries,
              causal shifting, individual-vs-batched equivalence, a second
              independently-implemented masked-loss scorer, zero-trainable-
              params assertion, no-gold-label-leakage structural check)
artifacts/    generated instruction candidates, selected prompt/demo configs,
              data manifest (row counts + file hashes, no raw text)
results/      screening results, final A-D metrics/predictions, final report,
              independent audit outputs
```

## Dataset access

Dataset CSVs (`data/*.csv`) are intentionally **not committed**:

- SST (source): the HF dataset card states `license: unknown`; Stanford's own
  release page states no formal license either.
- Yelp Review Full (target): distributed under the Yelp Dataset Agreement, a
  real research/non-commercial-style usage-restricted license.

To regenerate them exactly, run `src/prepare_data.py`, which pulls the
pinned HF revisions recorded in `configs/protocol.yaml` and reproduces the
same files (verifiable against the content hashes in
`artifacts/manifests/data_manifest.json`). Full schema, label mapping, and
license documentation: `data/DATASET_README.md`, `docs/DATA_VERIFICATION.md`.

## Reproduction

```bash
python -m venv .venv --system-site-packages
source .venv/bin/activate
PIP_CONSTRAINT= pip install -r requirements.txt

python src/prepare_data.py                 # regenerate data/*.csv
pytest tests/ -v                            # verify scoring interface before any inference
python src/run_screening.py                 # zero-shot screen -> selected_setting.json
python src/run_experiment.py                # resumable; skips any seed/condition already written
python scripts/audit_results.py             # independent metric recomputation
```

`run_experiment.py` is resumable: it skips any condition/seed whose
`results/experiment/metrics/{name}_final.json` already exists.
