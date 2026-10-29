"""Section 10/11: run conditions A-D across 3 seeds on the selected setting
(three_class, Qwen2.5-0.5B-Instruct), resumable at the stage level (skips any
condition/seed whose final JSON already exists) and at the scoring level (via
EvalCache, resumable across interrupted runs).

Usage: .venv/bin/python src/run_experiment.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_backend import load_frozen_model  # noqa: E402
from score_labels import score_candidate_labels_cached, predict_label  # noqa: E402
from metrics import accuracy, macro_f1, per_class_prf, confusion_matrix  # noqa: E402
from cache import EvalCache  # noqa: E402
from propose_instructions import propose_instructions  # noqa: E402
from search_prompts import sample_balanced_demos, generate_candidate_demo_sets, select_best_combination  # noqa: E402

RESULTS_DIR = ROOT / "results" / "experiment"
ARTIFACTS_DIR = ROOT / "artifacts"
FORMULATION = "three_class"
LABEL_COL = "label_name_3class"
SEEDS = [0, 1, 2]
D_TOP_K_INSTRUCTIONS = 3  # amendment, see configs/protocol.yaml d_search_budget_amendment


def load_protocol() -> dict:
    return yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())


def evaluate_config(lm, cache, instruction: str, demos, df: pd.DataFrame, label_names: list[str], max_input_tokens: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    for _, row in df.iterrows():
        scores = score_candidate_labels_cached(lm, cache, instruction, row["text"], label_names, max_input_tokens, demos=demos)
        pred = predict_label(scores)
        rows.append({"id": row["id"], "gold": row[LABEL_COL], "pred": pred})
    pred_df = pd.DataFrame(rows)
    y_true, y_pred = pred_df["gold"].tolist(), pred_df["pred"].tolist()
    metrics = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, label_names),
        "per_class": per_class_prf(y_true, y_pred, label_names),
        "confusion_matrix": {"class_order": label_names, "matrix": confusion_matrix(y_true, y_pred, label_names)},
        "n": len(df),
    }
    return pred_df, metrics


def save_final(name: str, instruction: str, demos, source_metrics, target_metrics, source_preds, target_preds, extra: dict | None = None):
    (RESULTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    source_preds.to_csv(RESULTS_DIR / "predictions" / f"{name}_source_test.csv", index=False)
    target_preds.to_csv(RESULTS_DIR / "predictions" / f"{name}_target_test.csv", index=False)
    payload = {
        "name": name,
        "instruction": instruction,
        "demos": demos,
        "source_test": source_metrics,
        "final_target_test": target_metrics,
        **(extra or {}),
    }
    (RESULTS_DIR / "metrics" / f"{name}_final.json").write_text(json.dumps(payload, indent=2, default=str))


def already_done(name: str) -> bool:
    return (RESULTS_DIR / "metrics" / f"{name}_final.json").exists()


def main() -> None:
    protocol = load_protocol()
    label_names = protocol["task_formulations"][1]["labels"]
    assert protocol["task_formulations"][1]["name"] == FORMULATION
    baseline_instruction = protocol["baseline_instruction"][FORMULATION].strip()
    demo_count = protocol["experiment_AD"]["demo_count"][FORMULATION]
    max_input_tokens = protocol["scoring"]["truncation"]["max_input_tokens"]
    n_inst_max = protocol["experiment_AD"]["instruction_candidates_max"]
    n_demo_sets_max = protocol["experiment_AD"]["demonstration_sets_max"]

    source_train = pd.read_csv(ROOT / "data" / "source_train.csv")
    source_validation = pd.read_csv(ROOT / "data" / "source_validation.csv")
    source_test = pd.read_csv(ROOT / "data" / "source_test.csv")
    final_target_test = pd.read_csv(ROOT / "data" / "final_target_test.csv")

    cache = EvalCache(ARTIFACTS_DIR / "cache" / "eval_cache.jsonl")
    lm = load_frozen_model("Qwen/Qwen2.5-0.5B-Instruct")
    n_trainable = sum(p.numel() for p in lm.model.parameters() if p.requires_grad)
    assert n_trainable == 0 and lm.model.training is False

    t_start = time.time()

    # ---------- Condition A: deterministic, computed once ----------
    if not already_done("A"):
        print("=== A: fixed baseline instruction, no demonstrations ===", flush=True)
        src_preds, src_metrics = evaluate_config(lm, cache, baseline_instruction, None, source_test, label_names, max_input_tokens)
        tgt_preds, tgt_metrics = evaluate_config(lm, cache, baseline_instruction, None, final_target_test, label_names, max_input_tokens)
        save_final("A", baseline_instruction, None, src_metrics, tgt_metrics, src_preds, tgt_preds, {"seed_invariant": True})
        print(f"  source_test acc={src_metrics['accuracy']:.4f}  target_test acc={tgt_metrics['accuracy']:.4f}", flush=True)
    else:
        print("=== A already done, skipping ===", flush=True)

    for seed in SEEDS:
        # ---------- Condition B: instruction search, no demos ----------
        inst_artifact_path = ARTIFACTS_DIR / "instructions" / f"seed{seed}.json"
        inst_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if inst_artifact_path.exists():
            candidates_scored = json.loads(inst_artifact_path.read_text())
        else:
            print(f"=== B seed={seed}: proposing instructions ===", flush=True)
            candidates = propose_instructions(
                lm, source_train, label_names, LABEL_COL, baseline_instruction, n_inst_max, seed
            )
            candidates_scored = []
            for cand in candidates:
                _, val_metrics = evaluate_config(lm, cache, cand["text"], None, source_validation, label_names, max_input_tokens)
                candidates_scored.append({**cand, "source_validation_macro_f1": val_metrics["macro_f1"]})
            inst_artifact_path.write_text(json.dumps(candidates_scored, indent=2, default=str))

        candidates_scored.sort(key=lambda c: (-c["source_validation_macro_f1"], c["index"]))
        best_inst = candidates_scored[0]

        name = f"B_seed{seed}"
        if not already_done(name):
            print(f"=== B seed={seed}: final eval, best_instruction_index={best_inst['index']} "
                  f"(val_macro_f1={best_inst['source_validation_macro_f1']:.4f}) ===", flush=True)
            src_preds, src_metrics = evaluate_config(lm, cache, best_inst["text"], None, source_test, label_names, max_input_tokens)
            tgt_preds, tgt_metrics = evaluate_config(lm, cache, best_inst["text"], None, final_target_test, label_names, max_input_tokens)
            save_final(name, best_inst["text"], None, src_metrics, tgt_metrics, src_preds, tgt_preds,
                       {"selected_instruction_index": best_inst["index"], "all_candidates": candidates_scored})
            print(f"  source_test acc={src_metrics['accuracy']:.4f}  target_test acc={tgt_metrics['accuracy']:.4f}", flush=True)
        else:
            print(f"=== B seed={seed} already done, skipping ===", flush=True)

        # ---------- Condition C: fixed instruction + random balanced demos ----------
        name = f"C_seed{seed}"
        c_demos = sample_balanced_demos(source_train, label_names, LABEL_COL, demo_count, seed=seed)
        if not already_done(name):
            print(f"=== C seed={seed}: fixed instruction + {demo_count} random balanced demos ===", flush=True)
            src_preds, src_metrics = evaluate_config(lm, cache, baseline_instruction, c_demos, source_test, label_names, max_input_tokens)
            tgt_preds, tgt_metrics = evaluate_config(lm, cache, baseline_instruction, c_demos, final_target_test, label_names, max_input_tokens)
            save_final(name, baseline_instruction, c_demos, src_metrics, tgt_metrics, src_preds, tgt_preds)
            print(f"  source_test acc={src_metrics['accuracy']:.4f}  target_test acc={tgt_metrics['accuracy']:.4f}", flush=True)
        else:
            print(f"=== C seed={seed} already done, skipping ===", flush=True)

        # ---------- Condition D: bounded joint search (amended budget) ----------
        search_artifact_path = ARTIFACTS_DIR / "selected_prompts" / f"D_seed{seed}_search.json"
        search_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        name = f"D_seed{seed}"
        if not already_done(name):
            if search_artifact_path.exists():
                search_result = json.loads(search_artifact_path.read_text())
            else:
                print(f"=== D seed={seed}: joint search (top-{D_TOP_K_INSTRUCTIONS} instructions x "
                      f"{n_demo_sets_max} demo sets) ===", flush=True)
                top_instructions = candidates_scored[:D_TOP_K_INSTRUCTIONS]
                demo_sets = generate_candidate_demo_sets(source_train, label_names, LABEL_COL, demo_count, n_demo_sets_max, base_seed=seed)
                search_result = select_best_combination(
                    lm, cache, top_instructions, demo_sets, source_validation, label_names, LABEL_COL, max_input_tokens
                )
                search_result["top_instructions"] = top_instructions
                search_result["demo_sets"] = demo_sets
                search_artifact_path.write_text(json.dumps(search_result, indent=2, default=str))

            best = search_result["best"]
            best_inst_d = next(c for c in search_result["top_instructions"] if c["index"] == best["instruction_index"])
            best_demos = search_result["demo_sets"][best["demo_set_index"]]
            print(f"  D seed={seed} best: instruction_index={best['instruction_index']} "
                  f"demo_set_index={best['demo_set_index']} val_macro_f1={best['macro_f1']:.4f}", flush=True)
            src_preds, src_metrics = evaluate_config(lm, cache, best_inst_d["text"], best_demos, source_test, label_names, max_input_tokens)
            tgt_preds, tgt_metrics = evaluate_config(lm, cache, best_inst_d["text"], best_demos, final_target_test, label_names, max_input_tokens)
            save_final(name, best_inst_d["text"], best_demos, src_metrics, tgt_metrics, src_preds, tgt_preds,
                       {"selected_instruction_index": best["instruction_index"], "selected_demo_set_index": best["demo_set_index"],
                        "source_validation_macro_f1": best["macro_f1"]})
            print(f"  source_test acc={src_metrics['accuracy']:.4f}  target_test acc={tgt_metrics['accuracy']:.4f}", flush=True)
        else:
            print(f"=== D seed={seed} already done, skipping ===", flush=True)

    elapsed = time.time() - t_start
    print(f"\n=== A-D EXPERIMENT COMPLETE in {elapsed/60:.1f} min (cache size={len(cache)}) ===")


if __name__ == "__main__":
    main()
