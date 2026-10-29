# Section 5: dataset verification (SST -> Yelp Review Full)

Verified 2026-10-28 by directly loading and inspecting real records (not from
documentation alone). All HF Hub access was live; `nlp.stanford.edu` (the
primary SST host) is unreachable from this sandboxed environment (`curl`:
"No route to host"), so the official archive could not be fetched directly --
see "SST provenance chain" below for how this was handled instead.

## Source: Stanford Sentiment Treebank (SST), sentence-level, continuous score

- HF repo: `stanfordnlp/sst`, config `default`.
- The repo ships only a Python *loading script* (`sst.py`), no data files.
  `datasets==5.1.0` refuses to execute it
  (`RuntimeError: Dataset scripts are no longer supported`) -- this is a
  `datasets`-library policy change (script execution removed in modern
  releases), **not** a license or access restriction.
- Resolved instead via Hugging Face's auto-generated Parquet mirror,
  branch `refs/convert/parquet`, commit `8a7d55ee01d70b8ae53e0b3c66a83b2a6198c2a7`.
  Files used: `default/train/0000.parquet`, `default/validation/0000.parquet`,
  `default/test/0000.parquet` (fetched directly with `pandas.read_parquet`,
  bypassing `datasets.load_dataset`'s config auto-detection, which has a
  demonstrated bug on this branch: it mis-casts the `ptb` config's data into
  the `default` schema and raises `CastError` if invoked naively).
- **Schema** (verified from real parquet data, not just the README):
  `sentence` (string), `label` (float32, confirmed range **[0.0, 1.0]** over
  all three splits), `tokens` (string), `tree` (string).
  **`tree` is NOT a sentiment label.** Inspected directly: it is a
  pipe-separated list of integers (e.g. `70|70|68|67|...|68`), i.e. CoreNLP
  binary-tree parent pointers, exactly the trap the brief warned about. It is
  not used anywhere in this project.
- **Splits** (sentence level, matches the paper's canonical sizes):
  train 8544, validation (dev) 1101, test 2210.
- This is the **sentence-level `default` config**, not the **239,232-row
  `dictionary` config** (every sub-phrase of every parse tree treated as its
  own example). We use only complete sentences, per the brief's requirement.
- **Citation**: Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C.D.,
  Ng, A., Potts, C. (2013). *Recursive Deep Models for Semantic
  Compositionality Over a Sentiment Treebank.* EMNLP 2013.
  https://www.aclweb.org/anthology/D13-1170/
- **License**: HF metadata states `license: unknown`. Stanford's original
  release page (https://nlp.stanford.edu/sentiment/) states no formal license
  terms either (page unreachable from this environment to re-confirm directly,
  but `dataset_info()`'s own `license:unknown` tag is taken as the accurate,
  honestly-reported status). Treated here as **unclear/unstated license,
  research-use dataset**, consistent with its ubiquitous use in academic NLP
  benchmarks; this is not a claim of any specific permissive license.
- **Original IDs**: the HF mirror does not carry Stanford's original numeric
  `sentence_index` (from `datasetSentences.txt`) or phrase IDs. This project
  mints its own stable IDs (`sst_<split>_<row_index>`, 0-indexed row order
  within each split as loaded from the pinned parquet commit) and documents
  this as a known limitation -- original Stanford sentence indices are not
  recoverable without access to the primary archive (blocked by network).

### 5-class / 3-class thresholds -- empirically verified, not assumed

The brief requires verifying the continuous-to-discrete threshold rule rather
than assuming it. Procedure actually run (see
`scripts/verify_sst_thresholds.py` output captured below): the continuous
`default`-config train split (8544 rows) was joined by normalized sentence
text against the independently-maintained discrete 5-class mirror
`SetFit/sst5` (commit `e51bdcd8cd3a30da231967c1a249ba59361279a3`, train.jsonl,
8544 rows, 5 labels: very negative/negative/neutral/positive/very positive).
8220/8544 rows matched by exact normalized text (the remainder differ by
PTB-escaping/quote-normalization edge cases and were not used for this check).

Candidate threshold rule tested: `score <= 0.2` -> very negative,
`<= 0.4` -> negative, `<= 0.6` -> neutral, `<= 0.8` -> positive,
`> 0.8` -> very positive (the standard SST-5 convention reported across the
literature).

**Result: 8220/8220 matched rows agree (100%).** This independently confirms
both (a) the exact threshold boundaries, and (b) that `SetFit/sst5`'s
pre-discretized labels are a faithful, non-resampled discretization of the
same official sentence-level data (same sentences, correct buckets) rather
than an independently-collected or arbitrarily-split resource. Given this,
**this project computes 5-class and 3-class SST labels itself, directly from
the verified continuous `default`-config scores**, rather than depending on
the third-party discrete mirror at runtime -- `SetFit/sst5` was used only for
this one-time verification check.

- **5-class mapping** (verified above): very_negative=0, negative=1,
  neutral=2, positive=3, very_positive=4.
- **3-class mapping** (this project's own collapse, applied after the
  verified 5-class assignment): negative = {very_negative, negative},
  neutral = {neutral}, positive = {positive, very_positive}.

### SST class distribution (verified counts)

| Split | very_neg | neg | neutral | pos | very_pos | n |
|---|---:|---:|---:|---:|---:|---:|
| train | 1092 | 2218 | 1624 | 2322 | 1288 | 8544 |
| dev (-> source_validation) | 139 | 289 | 229 | 279 | 165 | 1101 |
| test (-> source_test) | 279 | 633 | 389 | 510 | 399 | 2210 |

3-class collapse (test split): negative 912 (41.3%), neutral 389 (17.6%),
positive 909 (41.1%). Majority-class baseline (test, 3-class) = 41.3%
(negative, barely ahead of positive); uniform-chance baseline = 33.3%.
5-class majority-class baseline (test) = negative, 633/2210 = 28.6%; uniform
chance = 20%.

## Target: Yelp Review Full

- HF repo: `Yelp/yelp_review_full` (main branch, commit
  `c1f9ee939b7d05667af864ee1cb066393154bf85`), config `yelp_review_full`
  (plain Parquet, no loading script).
- **Schema** (verified from real records): `label`
  (`ClassLabel(names=['1 star','2 star','3 stars','4 stars','5 stars'])`,
  i.e. int 0-4 where star rating = label + 1), `text` (string).
- **Splits**: train 650,000 rows, test 50,000 rows. **Verified exactly
  balanced**: 130,000/class in train, 10,000/class in test (directly counted,
  not assumed from the dataset card).
- **Citation**: Zhang, X., Zhao, J., LeCun, Y. (2015). *Character-level
  Convolutional Networks for Text Classification.* NeurIPS 2015.
  https://arxiv.org/abs/1509.01626
- **License**: distributed under the official Yelp Dataset Agreement
  (https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf),
  same license family already handled for Yelp Polarity in the prior project.
  This is a real usage-restricted license (research/non-commercial-style
  terms per Yelp's agreement), not public domain -- recorded accurately here,
  not glossed over.
- **Original IDs**: this Zhang-et-al.-processed release strips all Yelp
  metadata (no original review ID, business ID, user ID, or date). Documented
  limitation: group-aware separation by business/user/review-thread is not
  possible with this resource; disjointness between `target_dev` and
  `final_target_test` is enforced only at the level of (a) drawing from
  official, disjoint Yelp train vs. test splits respectively, and (b)
  normalized-text duplicate checks across the two sampled files. IDs minted as
  `yelp_<split>_<row_index>`.
- **5-class mapping**: label (0-4) used directly, renamed 1_star..5_star.
- **3-class mapping**: negative = {1,2 star}, neutral = {3 star},
  positive = {4,5 star}. Verified proportions (exactly balanced source):
  40% / 20% / 40%, matching the brief's expected collapse of a balanced
  5-way split, not equal thirds.

## Source/target domain and annotation differences (to disclose, not fix)

- **Domain**: SST is short, single-sentence movie-review snippets from Rotten
  Tomatoes (median sentence length far shorter than a review); Yelp Review
  Full is full, multi-sentence business reviews (restaurants, services, etc.),
  often several sentences to a paragraph.
- **Annotation method**: SST labels originate from crowdsourced continuous
  sentiment sliders per phrase/sentence (Socher et al.), later bucketed into 5
  discrete bins by fixed thresholds (verified above). Yelp Review Full labels
  are the reviewer's own 1-5 star rating attached to the review at write time
  -- a self-reported behavioral/purchase-satisfaction signal, not a
  third-party sentiment annotation. These are related but **not equivalent**
  constructs: a star rating conflates sentiment with other factors (value,
  service, expectations), while the SST score is a direct sentiment judgment
  of text content only.
- **Granularity**: both are 5-level here, but SST's bins were defined by
  thresholding a continuous human sentiment score, while Yelp's 5 levels are
  the original integer rating scale. Mapping SST-5 sentiment intensity onto
  Yelp's 5-star scale (or either's 3-class collapse) is a modeling assumption
  made explicit by the table below, not a claim that the two scales are
  psychometrically identical.

| SST label | Yelp stars | 3-class |
|---|---:|---|
| very_negative | 1 | negative |
| negative | 2 | negative |
| neutral | 3 | neutral |
| positive | 4 | positive |
| very_positive | 5 | positive |

## Two frozen task formulations (Section 5/6)

1. **5-class**: SST-5 sentiment intensity -> Yelp 5-star rating, label sets
   verbatim as above.
2. **3-class**: SST 3-class collapse -> Yelp 3-class collapse (negative /
   neutral / positive), using the same underlying rows and the same
   verified per-row 5-class assignment, just a coarser label view.

Both formulations share the *same* underlying SST and Yelp rows (dev/test
samples drawn once; see `data/DATASET_README.md`), so no additional dataset
source or re-verification was required for the second formulation -- it is a
different label view of data already verified above, consistent with the
brief's "prefer the same underlying development examples across task
formulations where feasible."

No contamination claim is made in either direction: these are public,
widely-used benchmark datasets, and strong zero-shot performance (or its
absence) is not by itself evidence for or against pretraining exposure.
