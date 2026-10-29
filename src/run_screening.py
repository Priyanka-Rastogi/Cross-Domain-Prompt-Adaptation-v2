"""Section 9: run the frozen zero-shot screen (condition A only) over the
predefined candidate grid on target_dev. Never touches final_target_test.

Usage: .venv/bin/python src/run_screening.py
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
from score_labels import score_candidate_labels, predict_label  # noqa: E402
from metrics import accuracy, macro_f1, per_class_prf, confusion_matrix, majority_baseline  # noqa: E402

RESULTS_DIR = ROOT / "results" / "screening"


def run_one_setting(protocol: dict, formulation_name: str, model_id: str) -> dict:
    formulation = next(f for f in protocol["task_formulations"] if f["name"] == formulation_name)
    class_order = formulation["labels"]
    verbalizer_order = protocol["verbalizers"][formulation_name]["order"]
    verbalizer_text = protocol["verbalizers"][formulation_name]["text"]
    assert verbalizer_order == class_order
    instruction = protocol["baseline_instruction"][formulation_name].strip()
    max_input_tokens = protocol["scoring"]["truncation"]["max_input_tokens"]

    label_col = "label_name" if formulation_name == "five_class" else "label_name_3class"

    df = pd.read_csv(ROOT / "data" / "target_dev.csv")

    t0 = time.time()
    lm = load_frozen_model(model_id)
    load_time_s = time.time() - t0

    n_trainable = sum(p.numel() for p in lm.model.parameters() if p.requires_grad)
    assert n_trainable == 0
    assert lm.model.training is False

    predictions = []
    t1 = time.time()
    torch.cuda.reset_peak_memory_stats()
    for _, row in df.iterrows():
        scores = score_candidate_labels(lm, instruction, row["text"], verbalizer_text, max_input_tokens)
        pred_text = predict_label(scores)
        pred_label = class_order[verbalizer_text.index(pred_text)]
        predictions.append(
            {
                "id": row["id"],
                "gold": row[label_col],
                "pred": pred_label,
                **{f"mean_logprob_{s.label.replace(' ', '_')}": s.mean_logprob for s in scores},
            }
        )
    inference_time_s = time.time() - t1
    peak_mem_bytes = torch.cuda.max_memory_allocated()

    pred_df = pd.DataFrame(predictions)
    y_true = pred_df["gold"].tolist()
    y_pred = pred_df["pred"].tolist()

    acc = accuracy(y_true, y_pred)
    mf1 = macro_f1(y_true, y_pred, class_order)
    prf = per_class_prf(y_true, y_pred, class_order)
    cm = confusion_matrix(y_true, y_pred, class_order)
    maj = majority_baseline(y_true)
    uniform = 1.0 / len(class_order)

    eligible_lo, eligible_hi = protocol["screening"]["eligibility_range"]
    eligible = eligible_lo <= acc <= eligible_hi

    out_dir = RESULTS_DIR / formulation_name / model_id.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "target_dev_predictions.csv", index=False)

    result = {
        "formulation": formulation_name,
        "model_id": model_id,
        "model_revision": lm.revision,
        "n_examples": len(df),
        "accuracy": acc,
        "macro_f1": mf1,
        "majority_baseline": maj,
        "uniform_baseline": uniform,
        "per_class": prf,
        "confusion_matrix": {"class_order": class_order, "matrix": cm},
        "eligible": eligible,
        "trainable_params": n_trainable,
        "load_time_s": load_time_s,
        "inference_time_s": inference_time_s,
        "peak_gpu_memory_bytes": peak_mem_bytes,
        "n_model_calls": len(df),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=str))

    del lm.model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    protocol = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for formulation in [f["name"] for f in protocol["task_formulations"]]:
        for ckpt in protocol["checkpoints"]:
            model_id = ckpt["id"]
            print(f"=== screening: formulation={formulation} model={model_id} ===", flush=True)
            res = run_one_setting(protocol, formulation, model_id)
            print(
                f"  accuracy={res['accuracy']:.4f} macro_f1={res['macro_f1']:.4f} "
                f"eligible={res['eligible']} inference_time_s={res['inference_time_s']:.1f}",
                flush=True,
            )
            all_results.append(res)

    (RESULTS_DIR / "screening_summary.json").write_text(json.dumps(all_results, indent=2, default=str))

    eligible = [r for r in all_results if r["eligible"]]
    print("\n=== SCREEN COMPLETE ===")
    for r in all_results:
        print(f"{r['formulation']:12s} {r['model_id']:32s} acc={r['accuracy']:.4f} eligible={r['eligible']}")

    if not eligible:
        print("\nNO SETTING IS ELIGIBLE (60-80% zero-shot accuracy). Per the frozen stopping rule, "
              "stop here and report. Do not expand the search.")
    else:
        # tie-break: smallest checkpoint, then five_class over three_class, then closest to 0.70
        def ckpt_size(model_id: str) -> float:
            return 0.5 if "0.5B" in model_id else (1.5 if "1.5B" in model_id else float("inf"))

        eligible.sort(key=lambda r: (ckpt_size(r["model_id"]), r["formulation"] != "five_class", abs(r["accuracy"] - 0.70)))
        chosen = eligible[0]
        print(f"\nSELECTED SETTING: formulation={chosen['formulation']} model={chosen['model_id']} "
              f"accuracy={chosen['accuracy']:.4f}")
        (RESULTS_DIR / "selected_setting.json").write_text(json.dumps(chosen, indent=2, default=str))


if __name__ == "__main__":
    main()
