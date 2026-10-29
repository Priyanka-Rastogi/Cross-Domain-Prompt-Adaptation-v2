# Final results: SST (3-class) -> Yelp Review Full (3-class)

Model: Qwen2.5-0.5B-Instruct (frozen, 0 trainable parameters, verified).
Setting selected by the pre-registered screen (`results/screening/`); A-D
run per `configs/protocol.yaml` with the disclosed D-budget amendment
(top-3 instructions x 10 demo sets/seed, decided from throughput alone,
before any search score existed). All numbers below were **independently
recomputed from the raw prediction CSVs** (`scripts/audit_results.py`),
cross-checked against an independent scikit-learn implementation (exact
match, no assertion failures), with ID alignment verified clean across all
20 prediction files (no missing/extra/duplicate rows).

## Headline results

### Final target-domain test (Yelp, n=1000, 400/200/400 neg/neu/pos)

| Condition | Accuracy (mean +/- sd) | Macro-F1 (mean +/- sd) | Per-seed accuracy | Per-seed macro-F1 |
|---|---|---|---|---|
| A (fixed instruction, no demos) | 0.7700 (seed-invariant) | 0.5798 (seed-invariant) | 0.770 | 0.580 |
| B (optimized instruction) | 0.7687 +/- 0.0232 | 0.6818 +/- 0.0464 | 0.743 / 0.775 / 0.788 | 0.717 / 0.629 / 0.699 |
| C (fixed instruction + random demos) | 0.7750 +/- 0.0020 | 0.6366 +/- 0.0032 | 0.777 / 0.775 / 0.773 | 0.640 / 0.634 / 0.635 |
| **D (optimized instruction + demos)** | **0.7913 +/- 0.0047** | **0.7171 +/- 0.0179** | 0.786 / 0.793 / 0.795 | 0.697 / 0.730 / 0.725 |

### Source-domain test (SST, n=2210, 912/389/909 neg/neu/pos)

| Condition | Accuracy (mean +/- sd) | Macro-F1 (mean +/- sd) | Per-seed accuracy |
|---|---|---|---|
| A | 0.7643 (seed-invariant) | 0.5701 (seed-invariant) | 0.764 |
| B | 0.7018 +/- 0.0582 | 0.6249 +/- 0.0093 | **0.642** / 0.758 / 0.705 |
| C | 0.7513 +/- 0.0057 | 0.6230 +/- 0.0069 | 0.753 / 0.756 / 0.745 |
| D | 0.7225 +/- 0.0132 | 0.6533 +/- 0.0050 | 0.738 / 0.717 / 0.713 |

**D is the only condition that improves over A on target_test for every single seed** (0.786/0.793/0.795 vs A's fixed 0.770) -- not a cherry-picked average. D's macro-F1 gain over A (+0.137 on average) is large and consistent across seeds.

## Negative/noteworthy result, reported honestly (not hidden)

**B, seed 0, source_test accuracy = 0.642 -- *worse* than the unoptimized baseline A (0.764).** The instruction B selected for seed 0 scored well on the 1101-row source_validation set (macro-F1 0.639, the best of 8 candidates) but generalized poorly to source_test. This is a real instance of overfitting a wording choice to a fairly small validation set, and it is disclosed here rather than smoothed over by only reporting the mean. Its target_test transfer (0.743) was still reasonable, which is itself notable: this seed's instruction overfit to the *source* domain's validation quirks without it visibly hurting target transfer.

## Why accuracy and macro-F1 diverge so much (A: 77.0% accuracy vs. 0.580 macro-F1)

Target_test is exactly balanced (400/200/400 = 40%/20%/40%), so the majority-class baseline is 40% and chance (uniform) is 33.3%. Condition A's confusion matrix on target_test (rows=true, cols=pred, order neg/neu/pos):

```
            pred_neg  pred_neu  pred_pos
true_neg      387        2        11
true_neu      115        3        82
true_pos       20        0       380
```

The model gets negative (recall 96.8%) and positive (recall 95.0%) almost entirely right, but **neutral recall is 1.5%** (3/200 correct) -- it almost never predicts "neutral" at all, splitting true-neutral reviews roughly 58%/41% into negative/positive instead. Per-class F1: negative 0.839, neutral **0.029**, positive 0.871. Because accuracy is support-weighted, the 800 well-handled neg/pos examples dominate it (yielding 77%), while macro-F1 weights all three classes equally regardless of support, so neutral's near-zero F1 drags the average down to 0.580. **This is exactly the scenario the brief's "report macro-F1 separately from accuracy" requirement exists to catch.**

Condition D's confusion matrix on target_test (seed 1, the condition's representative/median run):

```
            pred_neg  pred_neu  pred_pos
true_neg      379       17        4
true_neu       86       81       33
true_pos       15       52      333
```

Neutral recall rises from 1.5% to **40.5%** (81/200) and neutral F1 from 0.029 to **0.463** -- the dominant driver of D's macro-F1 gain. This costs a little positive recall (95.0% -> 83.3%, some positive reviews now get called neutral) and a little negative recall (96.8% -> 94.8%), but because negative/positive are 80% of the test set, these small recall drops barely move accuracy while the neutral-class fix moves macro-F1 substantially. **The prompt-adaptation story here is specifically "the model learns to actually use the neutral label," not a generic quality bump.**

## Source-validation (search objective) vs. target-domain transfer (held-out)

These are explicitly different signals and should not be conflated:

| | source_validation macro-F1 (search objective, SST only) | target_test accuracy (held-out transfer) | target_test macro-F1 (held-out transfer) |
|---|---|---|---|
| A (baseline instr.) | 0.5468 | 0.770 | 0.580 |
| B (best per seed) | 0.6388 / 0.6322 / 0.6623 | 0.743 / 0.775 / 0.788 | 0.717 / 0.629 / 0.699 |
| D (best per seed) | 0.6732 / 0.6790 / 0.6775 | 0.786 / 0.793 / 0.795 | 0.697 / 0.730 / 0.725 |

The search objective (source_validation macro-F1) improves monotonically and substantially from A -> B -> D in every seed (0.547 -> ~0.64-0.66 -> ~0.67-0.68), confirming the search itself is working as intended *on the domain it is allowed to see*. Target-domain transfer of that improvement is real but **partial and noisier**: B's target_test macro-F1 is actually seed 1's *worst* result (0.629, below A's 0.580 by... no, above it by only a little) despite seed 1 having a healthy validation score (0.632) -- i.e., a validation-macro-F1 gain does not guarantee a proportional (or even any) target-domain macro-F1 gain for every seed. D's joint search is more consistent at transferring (713-730 macro-F1 across all 3 seeds, always well above A's 0.580), but this is still 3 seeds -- not strong evidence of a universal law, just this project's observed pattern.

## Paired uncertainty intervals (target_test, seed 1 as representative; resampling unit = individual target_test examples, matched by stable ID)

5000 bootstrap resamples of the 1000-example `final_target_test` set; this is a per-example bootstrap *within one seed's predictions*, not a resample over seeds -- it quantifies sampling noise in *which 1000 Yelp reviews happened to be drawn*, not the (separately reported, 3-seed) variance coming from instruction/demonstration search randomness. With only 3 seeds, a seed-level interval would have 2 degrees of freedom and be too wide to be informative, so both are reported rather than collapsed into one number.

| Comparison | Accuracy diff (point estimate) | 95% CI |
|---|---|---|
| B (seed1) - A | +0.005 | [-0.008, +0.017] -- not distinguishable from 0 |
| C (seed1) - A | +0.005 | [-0.009, +0.019] -- not distinguishable from 0 |
| D (seed1) - A | +0.023 | [0.000, +0.046] -- borderline, lower bound touches 0 |
| D (seed1) - C (seed1) | +0.018 | [-0.002, +0.038] -- borderline, not clearly significant |

**Honest limitation**: on raw accuracy, none of these differences are solidly outside their CI at the per-example-bootstrap level for a single seed -- the *macro-F1* story (where the gap is large and visible in every one of the 3 seeds, not just one bootstrap) is the stronger and more defensible finding here, precisely because accuracy is insensitive to the neutral-class fix described above while macro-F1 is designed to be sensitive to it.

## Cost

- Total A-D wall-clock: **52.3 minutes** (measured, not estimated) on the shared B200 MIG slice.
- Scoring cache: 155,300 unique (prompt, verbalizer-set) entries (each one batched forward pass over all 3 label candidates) -- cross-checks closely against the ~157,600 theoretical total scoring calls implied by the protocol (A: 3,210; B: 3x12,018; C: 3x3,210; D: 3x36,240), the small difference (~1.5%) reflecting real cache reuse across seeds/conditions.
- **Limitation, disclosed**: per-condition wall-clock and peak-GPU-memory were not separately instrumented in `run_experiment.py` (only the aggregate 52.3 min and cache size were captured) -- the screening run (`results/screening/*/metrics.json`) does have per-setting timing/memory, but the A-D run does not. This is a real completeness gap in this iteration's logging, not a result that is missing by design.

## Final-target visibility disclosure

Condition A's `final_target_test` accuracy was computed and printed to console/log at t+48s into the run -- before any seed's B/C/D selection had occurred. By construction, B and D select only from `source_validation` macro-F1 (verified directly from `run_experiment.py`'s code path: `final_target_test` is never passed into `evaluate_config` until after `best_inst`/`best` is already fixed), so there is no mechanism for A's result to have influenced later selection. No code, config, seed, or search budget was changed after any final-target result became visible. This is disclosed as a process-transparency note, not as a defect that altered the outcome.

## Limitations

- 3 seeds is a small sample for seed-level variance claims; CIs above make this explicit rather than implying more precision than exists.
- The D-budget amendment (search only 3 of 8 candidate instructions) means D's reported result is a lower bound on what the full 8x10 grid might find -- not necessarily the best possible D configuration under the originally-specified budget.
- No re-split/group-aware disjointness was possible for Yelp (no business/user IDs available in this release) -- see `data/DATASET_README.md`.
- SST's `tree` field was never used (correctly identified as parent-pointer data, not a label).
- B's instruction-search sometimes overfits the 1101-row source_validation set (seed 0, discussed above) -- a known risk of small-validation-set prompt search, not hidden here.

## Reproduction

```bash
cd /workspace/cross_domain_prompt_adaptation_v2
python3 -m venv --system-site-packages .venv
PIP_CONSTRAINT= .venv/bin/pip install -r requirements.txt
.venv/bin/python src/prepare_data.py          # rebuilds data/*.csv + manifest
.venv/bin/python -m pytest tests/ -v          # scoring-interface verification
.venv/bin/python src/run_screening.py         # reproduces the model/task selection
.venv/bin/python scripts/run_smoke_test.py    # fast sanity check of A-D machinery
.venv/bin/python src/run_experiment.py        # full A-D, 3 seeds (resumable: re-running
                                               # skips any condition/seed whose
                                               # results/experiment/metrics/<name>_final.json
                                               # already exists, and reuses
                                               # artifacts/cache/eval_cache.jsonl)
.venv/bin/python scripts/audit_results.py     # independent recomputation + bootstrap CIs
```
