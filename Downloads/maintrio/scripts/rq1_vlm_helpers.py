#!/usr/bin/env python3
"""Helpers for RQ1 general-purpose VLM evaluation on astronaut health panels.

Tabular OSDR / GeneLab panels are rendered as feature heatmaps so vision-language
models can score them. Labels use a flight-phase proxy (preflight vs return /
inflight), which is NOT a clinical endpoint — record that in validity_warning.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

VALIDITY_WARNING = (
    "phase-proxy labels + heatmap-rendered tabular panels — exploratory only, "
    "not a clinical detection result"
)

# Prompts used by zero-shot / VQA models
POSITIVE_PROMPT = "an abnormal postflight astronaut blood biomarker panel heatmap"
NEGATIVE_PROMPT = "a healthy preflight astronaut blood biomarker panel heatmap"
VQA_QUESTION = (
    "This image is a heatmap of an astronaut blood biomarker panel. "
    "Does it show abnormal postflight physiological stress? "
    "Answer Yes or No only."
)


def phase_proxy_label(timepoint: Any) -> int:
    """1 = exposed / return-inflight proxy, 0 = preflight baseline.

    Conventions seen in Inspiration4 panels: L-* (launch-minus) preflight,
    R+* (return-plus) postflight, Flight / FD* inflight.
    """
    tp = str(timepoint).strip().upper()
    if tp.startswith("R+") or tp.startswith("R-"):
        return 1
    if tp.startswith("FLIGHT") or tp.startswith("FD") or tp in {"INFLIGHT", "MID-FLIGHT"}:
        return 1
    if tp.startswith("L-") or tp.startswith("L+") or tp.startswith("PRE"):
        return 0
    # Unknown → treat as preflight (conservative negative)
    return 0


def parse_yes_no(text: str) -> int | None:
    """Parse a free-form model reply into 1 (Yes), 0 (No), or None."""
    if text is None:
        return None
    cleaned = str(text).strip().lower()
    # Prefer leading token
    head = re.split(r"[\s,.:;!?]", cleaned, maxsplit=1)[0]
    if head in {"yes", "y", "true", "abnormal", "positive"}:
        return 1
    if head in {"no", "n", "false", "normal", "negative", "healthy"}:
        return 0
    if re.search(r"\byes\b", cleaned) and not re.search(r"\bno\b", cleaned):
        return 1
    if re.search(r"\bno\b", cleaned) and not re.search(r"\byes\b", cleaned):
        return 0
    return None


def compute_table1_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Accuracy, Precision, Recall, F1, AUROC, MCC for Table 1."""
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


def _numeric_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    meta = {"subject_id", "timepoint", "label"}
    cols = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("No numeric feature columns found in panel.")
    mat = df[cols].astype(float).to_numpy()
    # Column-wise z-score; constant cols → 0
    mu = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    mat = (mat - mu) / sd
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat, cols


def render_panel_heatmaps(
    panel_csv: Path | str,
    out_dir: Path | str,
    dataset_key: str,
    label_fn: Callable[[Any], int] | None = None,
    max_features: int = 64,
) -> list[dict[str, Any]]:
    """Render each panel row as a heatmap PNG; return manifest entries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panel_csv = Path(panel_csv)
    out_dir = Path(out_dir)
    img_dir = out_dir / "images" / dataset_key
    img_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(panel_csv)
    if "subject_id" not in df.columns or "timepoint" not in df.columns:
        raise ValueError(f"{panel_csv} must have subject_id and timepoint columns")

    label_fn = label_fn or phase_proxy_label
    mat, feature_names = _numeric_feature_matrix(df)
    if mat.shape[1] > max_features:
        # Keep highest-variance columns for a readable image
        var = mat.var(axis=0)
        keep = np.argsort(var)[-max_features:]
        keep = np.sort(keep)
        mat = mat[:, keep]
        feature_names = [feature_names[i] for i in keep]

    samples: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        subject = str(row["subject_id"])
        timepoint = str(row["timepoint"])
        label = int(label_fn(timepoint))
        sample_id = f"{dataset_key}_{subject}_{timepoint}_{i}".replace("/", "-")
        path = img_dir / f"{sample_id}.png"

        vec = mat[i]
        # Arrange as a roughly square grid for a proper "image"
        n = len(vec)
        side = int(np.ceil(np.sqrt(n)))
        grid = np.full(side * side, np.nan, dtype=float)
        grid[:n] = vec
        grid = grid.reshape(side, side)

        fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=64)
        ax.imshow(grid, cmap="coolwarm", vmin=-2.5, vmax=2.5, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{subject} {timepoint}", fontsize=8)
        fig.tight_layout(pad=0.2)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        rel = path  # absolute preferred for local; notebook may relativize
        samples.append({
            "id": sample_id,
            "image_path": str(path),
            "dataset": dataset_key,
            "subject_id": subject,
            "timepoint": timepoint,
            "label": label,
            "n_features": int(n),
        })

    manifest_path = out_dir / f"manifest_{dataset_key}.json"
    manifest_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    return samples


def resolve_panel_path(dataset_key: str, repo_root: Path | None = None) -> Path | None:
    """Find panel.csv under repo data/raw or /kaggle/input (any nesting)."""
    root = repo_root or REPO_ROOT
    candidates: list[Path] = [
        root / "data" / "raw" / dataset_key / "panel.csv",
        Path(f"/kaggle/input/{dataset_key}/panel.csv"),
        # Common Kaggle Dataset upload layouts
        Path(f"/kaggle/input/algodata/panel.csv"),
        Path("/kaggle/input/datasets/srinithyapaninga/algodata/panel.csv"),
    ]

    search_roots = [
        Path(f"/kaggle/input/{dataset_key}"),
        Path("/kaggle/input/algodata"),
        Path("/kaggle/input/datasets"),
        Path("/kaggle/input"),
    ]
    for base in search_roots:
        if base.is_dir():
            # Prefer paths whose parent folder name matches the key or "algodata"
            hits = sorted(base.rglob("panel.csv"))
            for hit in hits:
                parent = hit.parent.name.lower()
                if parent in {dataset_key.lower(), "algodata", "inspiration4"}:
                    candidates.append(hit)
            candidates.extend(hits)

    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.is_file():
            return path
    return None


def empty_table1_row(model: str, dataset: str, status: str = "missing_data") -> dict[str, Any]:
    return {
        "model": model,
        "dataset": dataset,
        "status": status,
        "accuracy": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "auroc": float("nan"),
        "mcc": float("nan"),
        "n_samples": 0,
        "validity_warning": VALIDITY_WARNING,
    }


def table1_row(
    model: str,
    dataset: str,
    metrics: dict[str, float],
    n_samples: int,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "model": model,
        "dataset": dataset,
        "status": status,
        "accuracy": round(metrics["accuracy"], 4) if metrics.get("accuracy") == metrics.get("accuracy") else float("nan"),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "auroc": round(metrics["auroc"], 4) if metrics.get("auroc") == metrics.get("auroc") else float("nan"),
        "mcc": round(metrics["mcc"], 4) if metrics.get("mcc") == metrics.get("mcc") else float("nan"),
        "n_samples": int(n_samples),
        "validity_warning": VALIDITY_WARNING,
    }
