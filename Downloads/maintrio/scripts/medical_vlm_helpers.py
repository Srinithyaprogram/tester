#!/usr/bin/env python3
"""Helpers for medical-domain VLM experiments (Med-Flamingo, CheXagent, BioViL-T).

Uses real chest X-ray images (PneumoniaMNIST / MIMIC-style CXR), NOT Inspiration4
tabular heatmaps — those are the wrong modality for CXR foundation models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

PNEUMONIA_QUESTION = (
    "Based on this chest X-ray image, is there visual evidence of pneumonia? "
    "Respond with either 'Yes' or 'No'."
)

BIOVIL_POS_PROMPTS = [
    "Findings consistent with pneumonia.",
    "Consolidation suggestive of pneumonia.",
    "There is pneumonia.",
]
BIOVIL_NEG_PROMPTS = [
    "No evidence of pneumonia.",
    "Lungs are clear.",
    "No focal consolidation.",
]

VALIDITY_NOTE = (
    "Evaluated on labeled chest X-rays (PneumoniaMNIST). "
    "Not a clinical deployment result."
)


def load_medical_benchmark(repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Load medical samples from data/benchmark/dataset.json (paths made absolute)."""
    root = repo_root or REPO_ROOT
    manifest = root / "data" / "benchmark" / "dataset.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Missing {manifest}. Run: python scripts/generate_benchmark_dataset.py"
        )
    samples = json.loads(manifest.read_text(encoding="utf-8"))
    medical = [s for s in samples if s.get("domain") == "medical"]
    out = []
    for s in medical:
        p = Path(s["image_path"])
        if not p.is_absolute():
            p = root / p
        item = dict(s)
        item["image_path"] = str(p)
        # Normalize label: Yes/No or 0/1
        gt = str(s.get("ground_truth", "")).strip().lower()
        if gt in {"yes", "1", "pneumonia", "true"}:
            item["label"] = 1
        elif gt in {"no", "0", "normal", "false"}:
            item["label"] = 0
        else:
            meta = s.get("metadata") or {}
            item["label"] = 1 if meta.get("label_name") == "pneumonia" else 0
        out.append(item)
    return out


def parse_yes_no(text: str) -> int | None:
    import re

    if text is None:
        return None
    cleaned = str(text).strip().lower()
    head = re.split(r"[\s,.:;!?]", cleaned, maxsplit=1)[0]
    if head in {"yes", "y", "true", "pneumonia", "positive", "abnormal"}:
        return 1
    if head in {"no", "n", "false", "normal", "negative", "clear"}:
        return 0
    if re.search(r"\byes\b", cleaned) and not re.search(r"\bno\b", cleaned):
        return 1
    if re.search(r"\bno\b", cleaned) and not re.search(r"\byes\b", cleaned):
        return 0
    return None


def compute_table_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int).ravel()

    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true.size else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if y_true.size else float("nan"),
    }
    if len(np.unique(y_true)) < 2:
        out["auroc"] = float("nan")
    else:
        try:
            out["auroc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["auroc"] = float("nan")
    return out


def table_row(
    model: str,
    dataset: str,
    metrics: dict[str, float] | None = None,
    n_samples: int = 0,
    status: str = "completed",
    note: str = VALIDITY_NOTE,
) -> dict[str, Any]:
    def _r(key: str) -> float:
        if not metrics or key not in metrics:
            return float("nan")
        v = metrics[key]
        return round(float(v), 4) if v == v else float("nan")

    return {
        "model": model,
        "dataset": dataset,
        "status": status,
        "accuracy": _r("accuracy"),
        "precision": _r("precision"),
        "recall": _r("recall"),
        "f1": _r("f1"),
        "auroc": _r("auroc"),
        "mcc": _r("mcc"),
        "n_samples": int(n_samples),
        "note": note,
    }
