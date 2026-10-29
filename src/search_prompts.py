"""Section 10 demonstration sampling + bounded instruction x demonstration
search (condition D), and the fixed demo-selection rule shared by C and D."""

from __future__ import annotations

import numpy as np


def sample_balanced_demos(source_train_df, label_names: list[str], label_col: str, demo_count: int, seed: int) -> list[tuple[str, str]]:
    """Label-interleaved balanced sample: cycle through classes in
    verbalizer order filling each class's quota before repeating (fixed
    ordering rule, configs/protocol.yaml experiment_AD.demo_token_budget_rule)."""
    rng = np.random.RandomState(seed)
    per_class = demo_count // len(label_names)
    remainder = demo_count - per_class * len(label_names)

    per_class_rows = {}
    for i, lab in enumerate(label_names):
        pool = source_train_df[source_train_df[label_col] == lab]
        take = per_class + (1 if i < remainder else 0)
        idx = rng.choice(pool.index.values, size=take, replace=False)
        per_class_rows[lab] = list(pool.loc[idx]["text"])

    ordered: list[tuple[str, str]] = []
    cursors = {lab: 0 for lab in label_names}
    while len(ordered) < demo_count:
        for lab in label_names:
            if cursors[lab] < len(per_class_rows[lab]):
                ordered.append((per_class_rows[lab][cursors[lab]], lab))
                cursors[lab] += 1
    return ordered[:demo_count]


def generate_candidate_demo_sets(
    source_train_df, label_names: list[str], label_col: str, demo_count: int, n_sets: int, base_seed: int
) -> list[list[tuple[str, str]]]:
    return [
        sample_balanced_demos(source_train_df, label_names, label_col, demo_count, seed=base_seed * 1000 + j)
        for j in range(n_sets)
    ]


def select_best_combination(
    loaded_model,
    cache,
    instructions: list[dict],
    demo_sets: list[list[tuple[str, str]]],
    source_validation_df,
    label_names: list[str],
    label_col: str,
    max_input_tokens: int,
) -> dict:
    """Evaluate every (instruction, demo_set) combination on source_validation
    macro-F1; return the winner with the full score table. Tie-break:
    earliest instruction index, then earliest demo-set index (both fixed
    before search, per configs/protocol.yaml selection_tie_break)."""
    from score_labels import score_candidate_labels_cached, predict_label
    from metrics import macro_f1

    results = []
    for inst_idx, inst in enumerate(instructions):
        for demo_idx, demos in enumerate(demo_sets):
            y_true, y_pred = [], []
            for _, row in source_validation_df.iterrows():
                scores = score_candidate_labels_cached(
                    loaded_model, cache, inst["text"], row["text"], label_names, max_input_tokens, demos=demos
                )
                y_true.append(row[label_col])
                y_pred.append(predict_label(scores))
            mf1 = macro_f1(y_true, y_pred, label_names)
            results.append({"instruction_index": inst["index"], "demo_set_index": demo_idx, "macro_f1": mf1})

    results.sort(key=lambda r: (-r["macro_f1"], r["instruction_index"], r["demo_set_index"]))
    best = results[0]
    return {"best": best, "all_results": results}
