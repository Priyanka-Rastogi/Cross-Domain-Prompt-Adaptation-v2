"""Accuracy / macro-F1 / per-class metrics / confusion matrix, kept deliberately
simple and independent of scikit-learn's own implementation so results can be
cross-checked against sklearn later if needed (Section 11 audit)."""

from __future__ import annotations

from collections import Counter


def confusion_matrix(y_true: list[str], y_pred: list[str], class_order: list[str]) -> list[list[int]]:
    idx = {c: i for i, c in enumerate(class_order)}
    cm = [[0] * len(class_order) for _ in class_order]
    for t, p in zip(y_true, y_pred):
        cm[idx[t]][idx[p]] += 1
    return cm


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    assert len(y_true) == len(y_pred) and len(y_true) > 0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def per_class_prf(y_true: list[str], y_pred: list[str], class_order: list[str]) -> dict:
    cm = confusion_matrix(y_true, y_pred, class_order)
    n = len(class_order)
    out = {}
    for i, c in enumerate(class_order):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp
        fn = sum(cm[i][r] for r in range(n)) - tp
        support = sum(cm[i])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[c] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return out


def macro_f1(y_true: list[str], y_pred: list[str], class_order: list[str]) -> float:
    prf = per_class_prf(y_true, y_pred, class_order)
    return sum(v["f1"] for v in prf.values()) / len(class_order)


def majority_baseline(y_true: list[str]) -> float:
    c = Counter(y_true)
    return max(c.values()) / len(y_true)
