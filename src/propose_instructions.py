"""Section 10 instruction proposal: the frozen model proposes alternative
instructions, grounded only in source_train examples and the baseline
instruction (source-only -- never shown target/Yelp text)."""

from __future__ import annotations

import torch


def _meta_prompt(baseline_instruction: str, label_names: list[str], examples: list[tuple[str, str]]) -> str:
    ex_block = "\n".join(f'- "{t}" -> {lab}' for t, lab in examples)
    return (
        "You are helping design a short instruction for a text-classification prompt.\n"
        f"The task: classify a review's sentiment as one of: {', '.join(label_names)}.\n"
        f"Current baseline instruction: \"{baseline_instruction}\"\n"
        f"A few labeled examples of the task (for grounding only):\n{ex_block}\n\n"
        "Write ONE alternative instruction sentence for this exact classification task. "
        "It must ask the reader to output exactly one of the listed labels. "
        "Do not mention Yelp, restaurants, or any domain -- keep it domain-neutral. "
        "Reply with only the instruction sentence, nothing else."
    )


def propose_instructions(
    loaded_model,
    source_train_df,
    label_names: list[str],
    label_col: str,
    baseline_instruction: str,
    n_candidates: int,
    seed: int,
    n_grounding_examples: int = 4,
) -> list[dict]:
    """Returns a list of {"index": int, "text": str, "source": "baseline"|"generated"}.
    Index 0 is always the frozen baseline instruction."""
    import numpy as np

    tokenizer = loaded_model.tokenizer
    model = loaded_model.model
    device = next(model.parameters()).device

    candidates = [{"index": 0, "text": baseline_instruction, "source": "baseline"}]
    seen_texts = {baseline_instruction.strip().lower()}

    rng = np.random.RandomState(seed)

    attempts = 0
    max_attempts = (n_candidates - 1) * 4
    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        grounding_idx = rng.choice(len(source_train_df), size=min(n_grounding_examples, len(source_train_df)), replace=False)
        examples = [(source_train_df.iloc[i]["text"][:200], source_train_df.iloc[i][label_col]) for i in grounding_idx]
        prompt = _meta_prompt(baseline_instruction, label_names, examples)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)

        torch.manual_seed(seed * 1000 + attempts)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        generated = generated.strip('"').strip()

        key = generated.lower()
        if not generated or key in seen_texts or len(generated) > 300:
            continue
        seen_texts.add(key)
        candidates.append({"index": len(candidates), "text": generated, "source": "generated"})

    return candidates
