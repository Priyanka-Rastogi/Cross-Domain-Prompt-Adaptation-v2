"""Supervisor audit, check 5: fixed, documented sample from the EXISTING
final_target_test.csv (the same data already evaluated), cache disabled,
re-scored three ways:
  (a) original score_candidate_labels (individual, batch_size=1)
  (b) original _batched_score, run as one single batch with the other
      sample rows (mixed lengths)
  (c) an INDEPENDENTLY written scorer (different code path: per-example,
      no custom padding/masking code, uses HF's own `labels=` masked loss)
and compared against the actual recorded prediction from the real
experiment run (results/experiment/predictions/A_target_test.csv), to check
for any cache staleness/corruption.

Also documents: three_class verbalizers are single-token on this tokenizer
(see console output), so a genuinely multi-token check is done separately
against the five_class verbalizers on the SAME checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_backend import load_frozen_model  # noqa: E402
from score_labels import (  # noqa: E402
    score_candidate_labels,
    _batched_score,
    build_prompt_text,
    _tokenize_prompt_and_answer,
    predict_label,
)

INSTRUCTION = "Read the following review and classify its sentiment as exactly one of: negative, neutral, or positive."
THREE_CLASS = ["negative", "neutral", "positive"]
FIVE_CLASS = ["very negative", "negative", "neutral", "positive", "very positive"]


def independent_masked_loss_scorer(lm, instruction, review_text, label_texts, max_input_tokens=512, demos=None):
    """Deliberately different implementation: no custom batching/padding code,
    one example at a time, HF's own `labels=`-based loss (not the gather-based
    log_softmax indexing used in score_labels.py)."""
    from score_labels import _truncate_review

    tokenizer = lm.tokenizer
    device = next(lm.model.parameters()).device
    truncated_review = _truncate_review(tokenizer, review_text, max_input_tokens)
    prompt_text = build_prompt_text(tokenizer, instruction, truncated_review, demos)

    results = {}
    for label_text in label_texts:
        full_ids, answer_ids, _ = _tokenize_prompt_and_answer(tokenizer, prompt_text, " " + label_text)
        split = len(full_ids) - len(answer_ids)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        labels = input_ids.clone()
        labels[0, :split] = -100
        with torch.inference_mode():
            out = lm.model(input_ids=input_ids, labels=labels)
        mean_nll = out.loss.item()
        results[label_text] = -mean_nll  # mean_logprob
    return results


def main():
    lm = load_frozen_model("Qwen/Qwen2.5-0.5B-Instruct")

    df = pd.read_csv(ROOT / "data" / "final_target_test.csv")
    # documented fixed sample: shortest, median, longest text per gold class (varied lengths, all 3 classes)
    df["text_len"] = df["text"].str.len()
    sample_ids = []
    for cls in THREE_CLASS:
        sub = df[df["label_name_3class"] == cls].sort_values("text_len")
        picks = [sub.iloc[0], sub.iloc[len(sub) // 2], sub.iloc[-1]]
        sample_ids.extend(p["id"] for p in picks)
    sample = df[df["id"].isin(sample_ids)].reset_index(drop=True)
    print("FIXED DOCUMENTED SAMPLE (ids):", sample["id"].tolist())
    print()

    actual_preds = pd.read_csv(ROOT / "results" / "experiment" / "predictions" / "A_target_test.csv").set_index("id")

    mismatches = []
    for _, row in sample.iterrows():
        rid, text, gold = row["id"], row["text"], row["label_name_3class"]

        # (a) individual, cache disabled (score_candidate_labels never touches the cache)
        scores_individual = score_candidate_labels(lm, INSTRUCTION, text, THREE_CLASS)
        ind = {s.label: s.mean_logprob for s in scores_individual}
        pred_individual = predict_label(scores_individual)

        # (b) batched: score this example together with the rest of the sample in ONE forward pass
        from score_labels import _truncate_review
        full_id_seqs, answer_spans, order = [], [], []
        for _, r2 in sample.iterrows():
            t2 = _truncate_review(lm.tokenizer, r2["text"], 512)
            pt2 = build_prompt_text(lm.tokenizer, INSTRUCTION, t2, None)
            for lab in THREE_CLASS:
                fid, aid, _ = _tokenize_prompt_and_answer(lm.tokenizer, pt2, " " + lab)
                full_id_seqs.append(fid)
                answer_spans.append((len(fid) - len(aid), len(fid)))
                order.append((r2["id"], lab))
        device = next(lm.model.parameters()).device
        batched_scores = _batched_score(lm.model, lm.tokenizer, full_id_seqs, answer_spans, device)
        batched_this_row = {lab: mlp for (bid, lab), (_, mlp) in zip(order, batched_scores) if bid == rid}
        pred_batched = max(batched_this_row, key=batched_this_row.get)

        # (c) independently-written masked-loss scorer
        indep = independent_masked_loss_scorer(lm, INSTRUCTION, text, THREE_CLASS)
        pred_indep = max(indep, key=indep.get)

        actual_pred = actual_preds.loc[rid, "pred"]

        max_diff_a_vs_b = max(abs(ind[l] - batched_this_row[l]) for l in THREE_CLASS)
        max_diff_a_vs_c = max(abs(ind[l] - indep[l]) for l in THREE_CLASS)

        row_mismatch = not (pred_individual == pred_batched == pred_indep == actual_pred)
        if row_mismatch:
            mismatches.append(rid)

        print(f"id={rid} gold={gold:8s} actual_recorded_pred={actual_pred:8s} "
              f"individual={pred_individual:8s} batched={pred_batched:8s} independent_scorer={pred_indep:8s} "
              f"max|individual-batched|={max_diff_a_vs_b:.2e} max|individual-independent|={max_diff_a_vs_c:.2e} "
              f"{'*** MISMATCH ***' if row_mismatch else 'OK'}")

    print()
    print("=== multi-token verbalizer check (five_class, SAME checkpoint) ===")
    probe_text = sample.iloc[0]["text"]
    scores5 = score_candidate_labels(lm, "Classify sentiment as very negative, negative, neutral, positive, or very positive.", probe_text, FIVE_CLASS)
    for s in scores5:
        print(f"  {s.label:15s} n_answer_tokens={s.n_answer_tokens} mean_logprob={s.mean_logprob:.4f} sum_logprob={s.sum_logprob:.4f}")
    assert any(s.n_answer_tokens > 1 for s in scores5), "expected at least one multi-token label in five_class check"

    print()
    print("OVERALL:", "*** MISMATCHES FOUND ***" if mismatches else "ALL CLEAN -- individual/batched/independent-scorer/actual-recorded predictions agree on every sampled row.")
    if mismatches:
        print("mismatched ids:", mismatches)


if __name__ == "__main__":
    main()
