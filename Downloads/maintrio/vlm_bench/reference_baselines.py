#!/usr/bin/env python3
"""
Non-VLM reference points for Tables 1, 3 and 4.

A zero-shot VLM number on its own is uninterpretable: 0.62 AUROC could mean the
model is picking up real physiology, or it could mean the task is trivial, or
that the task is impossible and 0.62 is noise on n=60. This gives every VLM row
a floor and a ceiling to be read against.

    floor    majority-class and stratified-random predictors
    control  deviation_magnitude -- a single scalar, the mean absolute z-score
             of the sample. No fitting, no features selected, no model. Any VLM
             that does not beat this is detecting "something moved", not what
             moved. It exists because an earlier baseline-relative rendering
             scored 0.98-1.00 AUROC on this scalar alone; see
             vlm_bench/astro_modalities.py.
    ceiling  logistic regression and a gradient-boosted tree on the raw
             measurements, leave-one-subject-out

Leave-one-subject-out is the only defensible protocol at these cohort sizes
(4-10 subjects). Anything that puts the same crew member in train and test
inflates every metric -- see README "Guardrails".

    python vlm_bench/reference_baselines.py                  # all corpora
    python vlm_bench/reference_baselines.py --dataset inspiration4
    python vlm_bench/reference_baselines.py --label-col label_acute

Writes vlm_bench/results/reference_baselines.csv.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
RESULTS = Path(__file__).resolve().parent / "results"

#: Corpora with a real, non-circular label. alsda_physiology is deliberately
#: absent: its panel.csv is fabricated by make_example_panel.py.
DATASETS = ("inspiration4", "genelab_astronaut", "osdr_bedrest")

METADATA_COLUMNS = {
    "subject_id", "timepoint", "phase", "phase_source", "label", "label_acute",
    "split", "sample_id", "accession", "sex", "treatment",
}


def load_panel(dataset: str, label_col: str = "label") -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, subject_ids, feature_names) for one corpus."""
    path = RAW_ROOT / dataset / "panel.csv"
    if not path.exists():
        raise FileNotFoundError(f"no panel for '{dataset}' at {path}")
    frame = pd.read_csv(path, low_memory=False)
    if label_col not in frame.columns:
        raise ValueError(f"'{dataset}' has no '{label_col}' column")

    frame = frame[frame[label_col] >= 0].reset_index(drop=True)  # drops recovery rows
    features = [
        c for c in frame.columns
        if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(frame[c])
    ]
    x = frame[features].to_numpy(dtype=float)
    y = frame[label_col].to_numpy(dtype=int)
    groups = frame["subject_id"].to_numpy()
    return x, y, groups, features


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
        precision_score, recall_score, roc_auc_score,
    )

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0,
    }
    out["auroc"] = (
        roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    )
    out.update({
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
    })
    return {k: float(v) for k, v in out.items()}


def build_models(seed: int) -> dict[str, object]:
    return {
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "stratified_random": DummyClassifier(strategy="stratified", random_state=seed),
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            # Strong L2 is not optional at ~1000 features vs ~50 samples.
            LogisticRegression(C=0.01, max_iter=5000, random_state=seed),
        ),
        "gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=200, max_depth=3, learning_rate=0.1, random_state=seed
            ),
        ),
    }


def magnitude_control(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict from the mean absolute cohort z-score alone -- one number per sample.

    Needs no training, so there is nothing to cross-validate: the score is a
    fixed function of the row. Thresholded at its own median so the prediction
    is not trivially all-one-class.
    """
    std = x.std(axis=0, keepdims=True)
    z = (x - x.mean(axis=0, keepdims=True)) / np.where(std < 1e-8, 1.0, std)
    score = np.abs(np.nan_to_num(z)).mean(axis=1)
    return (score >= np.median(score)).astype(int), score


def evaluate(dataset: str, label_col: str, seed: int) -> list[dict]:
    x, y, groups, features = load_panel(dataset, label_col)
    n_subjects = len(np.unique(groups))
    rows: list[dict] = []

    pred, score = magnitude_control(x, y)
    rows.append({
        "model": "deviation_magnitude",
        "dataset": dataset,
        "protocol": "no fitting (single scalar)",
        "label_col": label_col,
        "n_samples": len(y),
        "n_subjects": n_subjects,
        "n_features": 1,
        "positive_rate": float(y.mean()),
        **_metrics(y, pred, score),
    })

    for name, model in build_models(seed).items():
        # Leave-one-subject-out: every sample is predicted by a model that never
        # saw that crew member.
        oof_pred = np.zeros(len(y), dtype=int)
        oof_score = np.zeros(len(y), dtype=float)
        splitter = LeaveOneGroupOut()
        for train_idx, test_idx in splitter.split(x, y, groups):
            if len(np.unique(y[train_idx])) < 2:
                # A held-out subject whose removal leaves one class: predict the
                # only class available rather than crashing the fold.
                only = int(y[train_idx][0])
                oof_pred[test_idx] = only
                oof_score[test_idx] = float(only)
                continue
            from sklearn.base import clone

            fitted = clone(model).fit(x[train_idx], y[train_idx])
            oof_pred[test_idx] = fitted.predict(x[test_idx])
            if hasattr(fitted, "predict_proba"):
                oof_score[test_idx] = fitted.predict_proba(x[test_idx])[:, 1]
            else:
                oof_score[test_idx] = oof_pred[test_idx]

        rows.append({
            "model": name,
            "dataset": dataset,
            "protocol": f"leave-one-subject-out (k={n_subjects})",
            "label_col": label_col,
            "n_samples": len(y),
            "n_subjects": n_subjects,
            "n_features": len(features),
            "positive_rate": float(y.mean()),
            **_metrics(y, oof_pred, oof_score),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, help="default: all")
    parser.add_argument(
        "--label-col", default="label", choices=("label", "label_acute"),
        help="'label_acute' drops the ambiguous recovery timepoints",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore", category=UserWarning)
    wanted = [args.dataset] if args.dataset else list(DATASETS)

    rows: list[dict] = []
    for dataset in wanted:
        try:
            rows.extend(evaluate(dataset, args.label_col, args.seed))
        except FileNotFoundError as exc:
            print(f"  {dataset}: skipped ({exc})")

    if not rows:
        print("No corpora available.")
        return 1

    frame = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"reference_baselines_{args.label_col}.csv"
    frame.to_csv(out, index=False)

    show = ["model", "dataset", "n_samples", "n_subjects", "accuracy",
            "balanced_accuracy", "f1", "auroc", "mcc"]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(frame[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    print(
        "\nRead balanced_accuracy, not accuracy. A VLM row below the "
        "majority_class floor is not detecting anything; a VLM row near the "
        "logistic_regression ceiling is doing as well as the features allow."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
