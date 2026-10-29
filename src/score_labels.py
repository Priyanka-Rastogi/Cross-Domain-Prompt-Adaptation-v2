"""Complete-label conditional-likelihood scoring (Section 8).

Scores every permitted answer verbalizer under teacher forcing and returns
the argmax. Never receives or uses a gold label. Aggregation is
length-normalized mean log-probability per answer token (fixed in
configs/protocol.yaml; summed log-likelihood would unfairly penalize longer
verbalizers such as "very positive" vs "neutral").

Prompt/answer boundary handling: the prompt and each candidate answer are
NOT tokenized separately and concatenated (BPE merge boundaries can silently
change token identity at the join). Instead the full prompt+answer string is
tokenized once, and the split point is the longest common token-id prefix
with the prompt tokenized alone -- robust to any BPE boundary effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LabelScore:
    label: str
    sum_logprob: float
    mean_logprob: float
    n_answer_tokens: int
    clean_boundary: bool  # True iff the split point equals len(prompt_ids) exactly


def build_prompt_text(tokenizer, instruction: str, review_text: str, demos: list[tuple[str, str]] | None = None) -> str:
    """demos: ordered list of (demo_text, demo_label_verbalizer) pairs, rendered
    as alternating user/assistant turns before the final query (standard
    chat-format few-shot ICL). Order is the caller's responsibility (Section
    10's fixed ordering rule) -- this function does not reorder."""
    messages = [{"role": "system", "content": instruction}]
    for demo_text, demo_label in demos or []:
        messages.append({"role": "user", "content": demo_text})
        messages.append({"role": "assistant", "content": demo_label})
    messages.append({"role": "user", "content": review_text})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _tokenize_prompt_and_answer(tokenizer, prompt_text: str, answer_text: str) -> tuple[list[int], list[int], bool]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt_text + answer_text, add_special_tokens=False)["input_ids"]
    split = _longest_common_prefix_len(prompt_ids, full_ids)
    answer_ids = full_ids[split:]
    if len(answer_ids) == 0:
        raise ValueError(
            f"empty answer token span for answer_text={answer_text!r}; "
            f"prompt_ids len={len(prompt_ids)}, full_ids len={len(full_ids)}"
        )
    clean = split == len(prompt_ids)
    return full_ids, answer_ids, clean


def score_candidate_labels(
    loaded_model,
    instruction: str,
    review_text: str,
    label_texts: list[str],
    max_input_tokens: int = 512,
    demos: list[tuple[str, str]] | None = None,
) -> list[LabelScore]:
    """Score every candidate label for ONE example. No gold label is accepted."""
    tokenizer = loaded_model.tokenizer
    model = loaded_model.model
    device = next(model.parameters()).device

    truncated_review = _truncate_review(tokenizer, review_text, max_input_tokens)
    truncated_demos = (
        [(_truncate_review(tokenizer, t, max_input_tokens), lab) for t, lab in demos] if demos else None
    )
    prompt_text = build_prompt_text(tokenizer, instruction, truncated_review, truncated_demos)

    full_id_seqs = []
    answer_spans = []  # (start_idx_in_seq, end_idx_in_seq_exclusive)
    clean_flags = []
    for label_text in label_texts:
        full_ids, answer_ids, clean = _tokenize_prompt_and_answer(tokenizer, prompt_text, " " + label_text)
        split = len(full_ids) - len(answer_ids)
        full_id_seqs.append(full_ids)
        answer_spans.append((split, len(full_ids)))
        clean_flags.append(clean)

    scores = _batched_score(model, tokenizer, full_id_seqs, answer_spans, device)

    return [
        LabelScore(
            label=label_texts[i],
            sum_logprob=scores[i][0],
            mean_logprob=scores[i][1],
            n_answer_tokens=answer_spans[i][1] - answer_spans[i][0],
            clean_boundary=clean_flags[i],
        )
        for i in range(len(label_texts))
    ]


def _truncate_review(tokenizer, text: str, max_input_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_input_tokens:
        return text
    truncated_ids = ids[:max_input_tokens]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)


@torch.inference_mode()
def _batched_score(
    model,
    tokenizer,
    full_id_seqs: list[list[int]],
    answer_spans: list[tuple[int, int]],
    device,
) -> list[tuple[float, float]]:
    """Right-padded batched forward pass; returns (sum_logprob, mean_logprob) per sequence."""
    pad_id = tokenizer.pad_token_id
    max_len = max(len(s) for s in full_id_seqs)
    batch_size = len(full_id_seqs)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(full_id_seqs):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[i, : len(seq)] = 1

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (B, L, V)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    results = []
    for i, (start, end) in enumerate(answer_spans):
        # logits at position p-1 predict token at position p (causal shift)
        target_positions = list(range(start, end))
        pred_logit_positions = [p - 1 for p in target_positions]
        token_logprobs = log_probs[i, pred_logit_positions, input_ids[i, target_positions]]
        s = token_logprobs.sum().item()
        n = len(target_positions)
        results.append((s, s / n))
    return results


def predict_label(label_scores: list[LabelScore]) -> str:
    best = max(label_scores, key=lambda ls: ls.mean_logprob)
    return best.label


def score_candidate_labels_cached(
    loaded_model,
    cache,
    instruction: str,
    review_text: str,
    label_texts: list[str],
    max_input_tokens: int = 512,
    demos: list[tuple[str, str]] | None = None,
) -> list[LabelScore]:
    """Same contract as score_candidate_labels, but checks/fills an EvalCache
    keyed on (model revision, full rendered prompt text, verbalizer set,
    aggregation, truncation) first -- resumable across interrupted runs."""
    tokenizer = loaded_model.tokenizer
    truncated_review = _truncate_review(tokenizer, review_text, max_input_tokens)
    truncated_demos = (
        [(_truncate_review(tokenizer, t, max_input_tokens), lab) for t, lab in demos] if demos else None
    )
    prompt_text = build_prompt_text(tokenizer, instruction, truncated_review, truncated_demos)
    key = cache.make_key(
        model_id=loaded_model.model_id,
        model_revision=loaded_model.revision,
        prompt_text=prompt_text,
        verbalizers=label_texts,
        aggregation="length_normalized_mean_logprob",
        max_input_tokens=max_input_tokens,
    )
    cached = cache.get(key)
    if cached is not None:
        return [LabelScore(**row) for row in cached]

    scores = score_candidate_labels(loaded_model, instruction, review_text, label_texts, max_input_tokens, demos)
    cache.put(key, [s.__dict__ for s in scores])
    return scores
