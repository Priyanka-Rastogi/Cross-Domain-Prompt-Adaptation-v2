# Supervisor audit: independent verification of the A-D evaluation (77-79% accuracy)

Conducted 2026-10-29, in response to a suspected evaluation bug. **No code,
configuration, prediction, or metric file was modified.** All original files
under `configs/`, `src/`, `data/`, and `results/experiment/` were hashed
before this audit and re-verified byte-identical afterward
(`sha256sum -c` exit code 0 against `pre_audit_hashes.txt`). Every check below
was run fresh, writing only new files under `results/audit_supervisor/` and
`scripts/supervisor_audit_part*.py`. No prompts, search budgets, seeds, or
scoring settings were changed at any point, including after viewing results.

## Audit table

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1a | SST continuous score -> 5-class, boundary cases (0.2/0.4/0.6/0.8 exactly) | **PASS** | Inline recompute: `<=0.2`->very_neg, `<=0.4`->neg, `<=0.6`->neutral, `<=0.8`->pos, `>0.8`->very_pos, confirmed at and either side of every boundary; matches the independent SetFit/sst5 cross-check from Section 5 (100% agreement, n=8220) |
| 1b | Yelp label 0-4 -> star 1-5 -> 3-class | **PASS** | Inline recompute + cross-checked against 5 real rows of `final_target_test.csv` |
| 1c | Identical class order across verbalizers/scores/predictions/metrics | **PASS** | `protocol.yaml`: `task_formulations.three_class.labels == verbalizers.three_class.order == verbalizers.three_class.text` (literal list equality, asserted). `src/run_experiment.py` threads one `label_names` list through scoring, prediction, and every metric call -- no index translation exists for the three_class path (unlike the five_class path, which is not what produced the audited numbers) |
| 1d | All three answers (incl. neutral) actually scored | **PASS** | `score_candidate_labels` loops over all of `label_texts` (len 3); verified live (part 5: every sampled row's score table includes neutral) |
| 2 | Recompute every metric from prediction CSVs, join by stable ID (NOT the embedded 'gold' column), check missing/duplicate/extra rows, exact correct/total per seed/condition | **PASS** | `scripts/supervisor_audit_part2.py` -> `results/audit_supervisor/part2_rejoined_metrics.json`. All 20 condition/split files: 0 missing, 0 extra, 0 duplicate IDs, 0 mismatches between freshly-rejoined gold (from `data/*.csv`) and the embedded `gold` column. Exact counts, e.g. **A/target_test: 770/1000**, **D_seed1/target_test: 793/1000** |
| 3a | Confusion matrix, predicted-class counts | **PASS** | A/target_test: `[[387,2,11],[115,3,82],[20,0,380]]`; predicted-class counts `{negative:522, neutral:5, positive:473}` out of 1000 |
| 3b | A's neutral recall = 3/200 | **PASS -- confirmed exactly** | 200 true-neutral rows: 115->negative, **3->neutral**, 82->positive |
| 3c | Majority (40%) / uniform (33.33%) baseline, neither equals reported accuracy | **PASS** | target_test distribution 400/200/400 confirmed; majority=0.400, uniform=0.3333, both well below A's 0.770 and D's ~0.79 |
| 4a | Tokenization at prompt/answer boundary | **PASS** | `_tokenize_prompt_and_answer` uses longest-common-token-prefix (not naive concatenation); `test_token_boundary_clean_for_all_verbalizers` re-run live, clean for all 3 verbalizers |
| 4b | Every answer token scored; termination policy | **PASS** | `target_positions = range(start, end)` covers every answer token; no EOS ever appended/scored (code never adds one; `add_special_tokens=False` throughout) |
| 4c | Causal shift, attention masks, padding, answer masks | **PASS** | Code inspection: `pred_logit_positions = [p-1 for p in target_positions]` (correct shift); right-padding with `attention_mask`; answer positions explicitly indexed (equivalent to masking). `test_manual_logprob_matches_masked_model_loss` re-run live: matches HF's independent `labels=`-based loss to <1e-3 |
| 4d | Score direction / argmax | **PASS** | `predict_label` = `max(..., key=mean_logprob)`; higher (less negative) log-prob wins, correct direction |
| 4e | Consistency with frozen summed/normalized rule | **PASS** | `protocol.yaml` declares `length_normalized_mean_logprob`; `predict_label` uses `mean_logprob`; cache key literally stores that string |
| 4f | Zero trainable parameters | **PASS** | Asserted inside `run_experiment.py` itself before any scoring (the real run would have crashed otherwise); re-confirmed live in this audit and in `test_zero_trainable_params` |
| 4g | Gold label never enters inference | **PASS** | `score_candidate_labels` has no gold/label parameter (structurally, `test_no_gold_label_argument_exists`); `evaluate_config` passes only `row["text"]` into scoring, reads `row[LABEL_COL]` only afterward for bookkeeping |
| 5a | Fixed documented sample, cache disabled, individual vs batched vs independent masked-loss scorer vs actual recorded prediction | **PASS** | `scripts/supervisor_audit_part5.py`. 9 real `final_target_test` rows (IDs logged), varied lengths, all 3 gold classes. All 4 computations agree on **every** row's predicted label. Independent scorer matches individual scoring to exact floating-point identity (0.00e+00) |
| 5b | Multi-token verbalizer handling | **PASS, with disclosure** | The 3 verbalizers actually used (`negative`/`neutral`/`positive`) are **single-token** on this tokenizer (ids 8225/20628/6785) -- length-normalization had zero numeric effect on the audited run. Multi-token handling is real and tested via the five_class verbalizers on the same checkpoint (`very negative`=2 tokens, mean-vs-sum differ as expected: -4.58 vs -9.16) |
| 5c | Batched-scoring numerical magnitude | **DISCLOSED, not a failure** | Multi-review batched scoring (27-sequence batch) showed bf16 numerical drift up to 0.22 nats vs. per-example scoring -- larger than the original Section-8 test's 1e-2 threshold assumed. **This code path was never used to produce the audited results**: the real run always scores one example's 3 labels per call (batch size 3), which matches the independent scorer to exact identity. Recommend re-scoping that test's tolerance/scope in a future iteration (see Limitations) |
| 6 | Cache key composition + no cross-condition/seed/task/checkpoint mixing | **PASS** | Key = sha256(model_id, model_revision, full rendered prompt_text [includes instruction+demos+query], verbalizer list, aggregation, max_input_tokens). Scanned entire 155,300-row cache: **0 duplicate keys, 0 value collisions** -- no two different (condition, seed) combinations ever produced the same key with different values, because no two ever produced the same key at all |
| 7 | A-D row-level comparison; D's accuracy gain = real ID-level flips | **PASS** | `scripts/supervisor_audit_part7.py`. All 3 seeds: `corrected - newly_wrong` exactly equals the actual net change in #correct (e.g. seed0: 55-39=+16, 770+16=786 = D's actual count). 11-16% of predictions change per seed (real churn, not a rounding artifact) |
| 8a | Normalized-text overlap across ALL experimental roles | **PASS** | Only the 3 pre-existing SST-internal overlaps already disclosed in `data/DATASET_README.md` (train/validation: 1, train/test: 2); **zero** overlap involving any Yelp file or across source<->target |
| 8b | Demonstrations sourced only from source_train, never equal to query text | **PASS** | All 18 C/D demo sets (6 configs x seeds) verified: every demo text found in `source_train`, **zero** overlap with any `source_test`/`final_target_test` query ever evaluated |
| -- | Original files unmodified throughout this audit | **PASS** | `sha256sum -c` against pre-audit baseline: exit code 0, all files identical |

**No "NOT VERIFIED" items remain** -- every requested check was executed with evidence. One item is reported with a partial scope: per-class confusion-matrix detail was independently re-derived and spot-checked in full for conditions **A** and **D (seed 1)**; for B and C across all seeds, aggregate accuracy/macro-F1 were independently re-verified exactly (check 2) and D's row-level decomposition (check 7) further cross-validates the pipeline, but individual per-class confusion cells for every B/C seed were not separately re-inspected by hand in this pass (they are available in `results/experiment/metrics/{B,C}_seed*_final.json` and were produced by the same already-audited code path).

## Conclusion

**No evaluation bug was found.** The reported 77.0% (A) and ~79.1% (D) target-domain accuracy figures are reproduced exactly by an independent re-derivation from the raw prediction CSVs (ID-joined against the original data files, cross-checked with scikit-learn), and the underlying scoring mechanics (tokenization, causal shifting, masking, argmax, caching) hold up under direct code audit, live re-testing, and a fresh cache-disabled sample re-score against an independently-written scorer.

The supervisor's suspicion is understandable given the accuracy numbers alone, but the mechanism is now fully explained and not a bug: condition A's small model almost never predicts "neutral" (3/200 recall, 5/1000 total neutral predictions), so its 77% accuracy is driven almost entirely by strong negative/positive discrimination (96-97% recall each) on 80% of the balanced test set, while macro-F1 (0.580) correctly penalizes the neutral blind spot. D's joint prompt optimization measurably and consistently (across all 3 seeds, with real ID-level flips) improves neutral handling (recall 1.5%->40.5%) and overall macro-F1 (0.580->0.717), with a more modest but real accuracy gain. This is a genuine, if partial, adaptation effect -- not an artifact of a scoring or labeling bug.

## Limitations of this audit itself

- 3-seed-level variance claims remain statistically weak (as already disclosed in `results/FINAL_REPORT.md`); this audit does not add seeds.
- The multi-review batched-scoring code path (check 5c) is correctness-verified (no prediction flips) but shows larger bf16 numerical variation than the original test tolerance assumed; since it was never used to produce the audited results, this does not affect the reported numbers, but the test should be tightened or re-scoped.
- Per-class confusion detail for B/C was not hand-re-inspected seed-by-seed (see audit-table note above); only aggregate metrics and D's row-level flips were independently re-derived for those conditions.

## File index

- `scripts/supervisor_audit_part2.py`, `part5.py`, `part7.py`, `part8.py` -- all new, none modify existing files.
- `results/audit_supervisor/part2_rejoined_metrics.json`, `part5_output.log`, `part7_row_level_comparison.json`, `part8_overlap_and_leakage.json`.
- `/tmp/.../pre_audit_hashes.txt` (session scratchpad) -- immutability baseline, re-verified clean.
