"""Section 10 source-only smoke test: tiny-scale run of the full A-D pipeline
(1 seed, 3 instructions, 2 demo sets, 20-row source_validation subsample) to
catch bugs before the full, expensive search. Never touches final_target_test
labels for selection (only uses them the same way the real pipeline would:
final eval after freezing).

Usage: .venv/bin/python scripts/run_smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_backend import load_frozen_model  # noqa: E402
from cache import EvalCache  # noqa: E402
from propose_instructions import propose_instructions  # noqa: E402
from search_prompts import sample_balanced_demos, generate_candidate_demo_sets, select_best_combination  # noqa: E402
import run_experiment as exp  # noqa: E402

FORMULATION = "three_class"
LABEL_COL = "label_name_3class"


def main() -> None:
    protocol = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())
    label_names = protocol["task_formulations"][1]["labels"]
    baseline_instruction = protocol["baseline_instruction"][FORMULATION].strip()
    max_input_tokens = protocol["scoring"]["truncation"]["max_input_tokens"]

    source_train = pd.read_csv(ROOT / "data" / "source_train.csv")
    source_validation = pd.read_csv(ROOT / "data" / "source_validation.csv").sample(20, random_state=0).reset_index(drop=True)
    source_test = pd.read_csv(ROOT / "data" / "source_test.csv").sample(20, random_state=0).reset_index(drop=True)
    final_target_test = pd.read_csv(ROOT / "data" / "final_target_test.csv").sample(20, random_state=0).reset_index(drop=True)

    exp.RESULTS_DIR = ROOT / "results" / "smoke"
    exp.LABEL_COL = LABEL_COL
    cache = EvalCache(ROOT / "artifacts" / "smoke_cache" / "eval_cache.jsonl")

    lm = load_frozen_model("Qwen/Qwen2.5-0.5B-Instruct")
    n_trainable = sum(p.numel() for p in lm.model.parameters() if p.requires_grad)
    assert n_trainable == 0 and lm.model.training is False
    print("OK: model frozen, 0 trainable params")

    # A
    src_preds, src_metrics = exp.evaluate_config(lm, cache, baseline_instruction, None, source_test, label_names, max_input_tokens)
    tgt_preds, tgt_metrics = exp.evaluate_config(lm, cache, baseline_instruction, None, final_target_test, label_names, max_input_tokens)
    print(f"A: source_acc={src_metrics['accuracy']:.3f} target_acc={tgt_metrics['accuracy']:.3f}")
    assert set(src_preds["pred"]).issubset(set(label_names))

    # B
    candidates = propose_instructions(lm, source_train, label_names, LABEL_COL, baseline_instruction, n_candidates=3, seed=0)
    assert len(candidates) >= 1 and candidates[0]["source"] == "baseline"
    for cand in candidates:
        _, val_metrics = exp.evaluate_config(lm, cache, cand["text"], None, source_validation, label_names, max_input_tokens)
        print(f"B candidate idx={cand['index']} source={cand['source']} val_macro_f1={val_metrics['macro_f1']:.3f}")

    # C
    c_demos = sample_balanced_demos(source_train, label_names, LABEL_COL, demo_count=6, seed=0)
    assert len(c_demos) == 6
    _, c_metrics = exp.evaluate_config(lm, cache, baseline_instruction, c_demos, final_target_test, label_names, max_input_tokens)
    print(f"C: target_acc={c_metrics['accuracy']:.3f}")

    # D (tiny: 2 instructions x 2 demo sets)
    demo_sets = generate_candidate_demo_sets(source_train, label_names, LABEL_COL, demo_count=6, n_sets=2, base_seed=0)
    search_result = select_best_combination(lm, cache, candidates[:2], demo_sets, source_validation, label_names, LABEL_COL, max_input_tokens)
    print("D best combo:", search_result["best"])
    assert len(search_result["all_results"]) == 2 * 2

    print("\nSMOKE TEST PASSED: A/B/C/D pipeline runs end-to-end without error.")


if __name__ == "__main__":
    main()
