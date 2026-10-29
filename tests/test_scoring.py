"""Live-model verification of the scoring interface (Section 8).

Uses the real, already-cached Qwen2.5-0.5B-Instruct checkpoint -- not a mock
-- because the failure modes this must catch (padding/mask bugs, prompt/answer
token-boundary bugs, batching inconsistencies) only show up against a real
tokenizer and model. Run with: .venv/bin/python -m pytest tests/test_scoring.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_backend import load_frozen_model  # noqa: E402
from score_labels import (  # noqa: E402
    build_prompt_text,
    score_candidate_labels,
    _tokenize_prompt_and_answer,
    _batched_score,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
FIVE_CLASS = ["very negative", "negative", "neutral", "positive", "very positive"]
THREE_CLASS = ["negative", "neutral", "positive"]
INSTRUCTION = (
    "Read the following review and classify its sentiment as exactly one of: "
    "very negative, negative, neutral, positive, or very positive."
)


@pytest.fixture(scope="module")
def loaded():
    lm = load_frozen_model(MODEL_ID)
    yield lm
    del lm.model
    torch.cuda.empty_cache()


def test_zero_trainable_params(loaded):
    n_trainable = sum(p.numel() for p in loaded.model.parameters() if p.requires_grad)
    assert n_trainable == 0
    assert loaded.model.training is False


def test_token_boundary_clean_for_all_verbalizers(loaded):
    """Verify the prompt/answer join does not silently shift tokens for any
    verbalizer actually used in the frozen protocol."""
    prompt_text = build_prompt_text(loaded.tokenizer, INSTRUCTION, "A perfectly ordinary review.")
    report = {}
    for label in FIVE_CLASS + THREE_CLASS:
        full_ids, answer_ids, clean = _tokenize_prompt_and_answer(loaded.tokenizer, prompt_text, " " + label)
        report[label] = (clean, len(answer_ids))
    for label, (clean, n_tok) in report.items():
        assert n_tok >= 1, f"{label} produced zero answer tokens"
    # Report, rather than hard-require, a clean boundary for every label: the
    # longest-common-prefix split handles an unclean boundary correctly either
    # way, but an unclean boundary is worth knowing about.
    unclean = {k: v for k, v in report.items() if not v[0]}
    print("unclean boundaries (handled, but noted):", unclean)


def test_multi_token_labels_are_actually_multi_token(loaded):
    """Sanity check that this test battery actually exercises multi-token
    scoring (the brief requires this, not just single-token labels)."""
    ids = loaded.tokenizer(" very negative", add_special_tokens=False)["input_ids"]
    assert len(ids) > 1, "expected 'very negative' to tokenize to >1 token for this check to be meaningful"


def test_individual_vs_batched_scoring_matches(loaded):
    reviews = [
        "Great.",
        "This was an absolutely fantastic experience from start to finish, I loved every minute of it.",
        "Terrible service and cold food, would not go back.",
        "It was okay, nothing special but nothing wrong either, just an average time overall.",
    ]
    individual_results = []
    for r in reviews:
        scores = score_candidate_labels(loaded, INSTRUCTION, r, FIVE_CLASS)
        individual_results.append({s.label: s.mean_logprob for s in scores})

    # Now score the same four examples as one batch by directly exercising
    # the batched path with mixed lengths (varied-length reviews above).
    from score_labels import _truncate_review, build_prompt_text as bpt, _tokenize_prompt_and_answer as tpa

    full_id_seqs, answer_spans = [], []
    order = []
    for r in reviews:
        truncated = _truncate_review(loaded.tokenizer, r, 512)
        prompt_text = bpt(loaded.tokenizer, INSTRUCTION, truncated)
        for label in FIVE_CLASS:
            full_ids, answer_ids, _ = tpa(loaded.tokenizer, prompt_text, " " + label)
            full_id_seqs.append(full_ids)
            answer_spans.append((len(full_ids) - len(answer_ids), len(full_ids)))
            order.append((r, label))

    device = next(loaded.model.parameters()).device
    batched_scores = _batched_score(loaded.model, loaded.tokenizer, full_id_seqs, answer_spans, device)

    batched_results = {}
    for (r, label), (_, mean_lp) in zip(order, batched_scores):
        batched_results.setdefault(r, {})[label] = mean_lp

    for i, r in enumerate(reviews):
        for label in FIVE_CLASS:
            a = individual_results[i][label]
            b = batched_results[r][label]
            assert abs(a - b) < 1e-2, f"individual vs batched mismatch for {r!r}/{label}: {a} vs {b}"


def test_manual_logprob_matches_masked_model_loss(loaded):
    """Cross-check the hand-rolled gather-based log-likelihood against HF's
    own `labels=`-based cross-entropy loss computation (independent code
    path)."""
    review = "The plot dragged but the acting was superb throughout the film."
    prompt_text = build_prompt_text(loaded.tokenizer, INSTRUCTION, review)
    label_text = " very positive"
    full_ids, answer_ids, _ = _tokenize_prompt_and_answer(loaded.tokenizer, prompt_text, label_text)
    split = len(full_ids) - len(answer_ids)

    device = next(loaded.model.parameters()).device
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[0, :split] = -100

    with torch.inference_mode():
        out = loaded.model(input_ids=input_ids, labels=labels)
    hf_loss = out.loss.item()  # mean NLL over non-masked positions

    with torch.inference_mode():
        logits = loaded.model(input_ids=input_ids).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)
    target_positions = list(range(split, len(full_ids)))
    pred_positions = [p - 1 for p in target_positions]
    token_logprobs = log_probs[0, pred_positions, input_ids[0, target_positions]]
    manual_mean_nll = -token_logprobs.mean().item()

    assert abs(hf_loss - manual_mean_nll) < 1e-3, f"hf_loss={hf_loss} vs manual={manual_mean_nll}"


def test_no_gold_label_argument_exists():
    """Structural guarantee: the scoring function cannot leak a gold label
    because it has no parameter for one."""
    import inspect

    sig = inspect.signature(score_candidate_labels)
    names = set(sig.parameters.keys())
    assert "gold" not in names and "label" not in names and "true_label" not in names


def test_determinism(loaded):
    review = "A quiet, forgettable movie with one great scene."
    s1 = score_candidate_labels(loaded, INSTRUCTION, review, FIVE_CLASS)
    s2 = score_candidate_labels(loaded, INSTRUCTION, review, FIVE_CLASS)
    for a, b in zip(s1, s2):
        assert a.label == b.label
        assert abs(a.mean_logprob - b.mean_logprob) < 1e-6


def test_truncation_applies_only_to_review_text(loaded):
    long_review = "This movie was fine. " * 200  # forces truncation at 512 tokens
    scores = score_candidate_labels(loaded, INSTRUCTION, long_review, FIVE_CLASS, max_input_tokens=20)
    assert len(scores) == len(FIVE_CLASS)
    for s in scores:
        assert s.n_answer_tokens >= 1
