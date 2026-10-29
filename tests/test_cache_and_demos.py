from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_backend import load_frozen_model  # noqa: E402
from score_labels import score_candidate_labels, score_candidate_labels_cached  # noqa: E402
from cache import EvalCache  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
THREE_CLASS = ["negative", "neutral", "positive"]
INSTRUCTION = "Read the following review and classify its sentiment as exactly one of: negative, neutral, or positive."


@pytest.fixture(scope="module")
def loaded():
    lm = load_frozen_model(MODEL_ID)
    yield lm
    del lm.model
    torch.cuda.empty_cache()


def test_demos_change_the_prediction_distribution(loaded):
    review = "The food came out lukewarm and the wait was long."
    demos = [
        ("Best meal of my life, absolutely perfect in every way.", "positive"),
        ("Disgusting, rude staff, never coming back.", "negative"),
        ("It was fine, nothing memorable either way.", "neutral"),
    ]
    no_demo = score_candidate_labels(loaded, INSTRUCTION, review, THREE_CLASS)
    with_demo = score_candidate_labels(loaded, INSTRUCTION, review, THREE_CLASS, demos=demos)
    no_demo_scores = {s.label: s.mean_logprob for s in no_demo}
    with_demo_scores = {s.label: s.mean_logprob for s in with_demo}
    assert no_demo_scores != with_demo_scores  # demos must actually change the prompt/scores


def test_cache_hit_matches_live_score_and_skips_recompute(tmp_path, loaded):
    cache = EvalCache(tmp_path / "eval_cache.jsonl")
    review = "A completely average experience, nothing more to say."

    live = score_candidate_labels(loaded, INSTRUCTION, review, THREE_CLASS)
    cached_first = score_candidate_labels_cached(loaded, cache, INSTRUCTION, review, THREE_CLASS)
    assert len(cache) == 1
    for a, b in zip(live, cached_first):
        assert abs(a.mean_logprob - b.mean_logprob) < 1e-4

    # second call must hit the cache: corrupt the model so a live recompute would error/differ
    original_forward = loaded.model.forward
    def _boom(*a, **k):
        raise RuntimeError("cache miss should not re-invoke the model")
    loaded.model.forward = _boom
    try:
        cached_second = score_candidate_labels_cached(loaded, cache, INSTRUCTION, review, THREE_CLASS)
    finally:
        loaded.model.forward = original_forward
    for a, b in zip(cached_first, cached_second):
        assert a.mean_logprob == b.mean_logprob


def test_cache_persists_across_instances(tmp_path, loaded):
    path = tmp_path / "eval_cache.jsonl"
    review = "Service was quick and the staff were friendly."
    cache1 = EvalCache(path)
    score_candidate_labels_cached(loaded, cache1, INSTRUCTION, review, THREE_CLASS)

    cache2 = EvalCache(path)  # reload from disk, simulating a resumed run
    assert len(cache2) == 1
