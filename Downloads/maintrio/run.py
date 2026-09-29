#!/usr/bin/env python3
"""
Fit and evaluate crew deterioration detectors.

    python run.py                                    # all classical models + MLP, on the synthetic smoke test
    python run.py --dataset inspiration4              # a real NASA panel (needs data/raw/inspiration4/panel.csv)
    python run.py --models gait_logreg,descriptor_mlp --seed 7

Balanced accuracy is the primary metric: deterioration is the minority class,
and a monitor that never alerts would otherwise look good. Splits are
subject-wise by default -- no crew member appears in both train and test.

NEVER report numbers from synthetic_crew_gait. It is a simulator tuned to land
around 0.80-0.88 balanced accuracy so pipeline breakage is visible, not a result.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

import datasets

MODEL_KEYS = (
    "prevalence_prior",
    "gait_logreg",
    "gait_random_forest",
    "gait_gradient_boosting",
    "descriptor_mlp",
)
PRIMARY_METRIC = "balanced_accuracy"

# --------------------------------------------------------------------------
# Descriptors: what the classical models and the MLP both train on
# --------------------------------------------------------------------------

FEATURE_SUFFIXES = (
    "mean", "std", "min", "max", "rms", "iqr",
    "zero_cross_rate", "dominant_freq", "spectral_entropy", "jerk_rms",
)


def window_features(x: np.ndarray, sampling_rate_hz: float = 50.0) -> np.ndarray:
    """Interpretable per-channel descriptors (cadence proxy, sway, jerk, ...)."""
    if x.ndim == 2:  # already engineered
        return x.astype(np.float32)

    n, c, t = x.shape
    eps = 1e-8

    mean = x.mean(axis=2)
    std = x.std(axis=2)
    minimum = x.min(axis=2)
    maximum = x.max(axis=2)
    rms = np.sqrt((x ** 2).mean(axis=2))
    q75, q25 = np.percentile(x, [75, 25], axis=2)
    iqr = q75 - q25

    centred = x - mean[:, :, None]
    zero_cross = np.mean(np.abs(np.diff(np.sign(centred), axis=2)) > 0, axis=2)

    spectrum = np.abs(np.fft.rfft(centred, axis=2)) ** 2
    freqs = np.fft.rfftfreq(t, d=1.0 / sampling_rate_hz)
    dominant = freqs[np.argmax(spectrum, axis=2)]
    power = spectrum / (spectrum.sum(axis=2, keepdims=True) + eps)
    spectral_entropy = -(power * np.log(power + eps)).sum(axis=2) / np.log(power.shape[2] + eps)

    jerk = np.diff(x, axis=2) * sampling_rate_hz
    jerk_rms = np.sqrt((jerk ** 2).mean(axis=2))

    stacked = np.stack(
        [mean, std, minimum, maximum, rms, iqr, zero_cross, dominant, spectral_entropy, jerk_rms],
        axis=2,
    )
    return stacked.reshape(n, c * stacked.shape[2]).astype(np.float32)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def count_parameters(model: Any) -> int:
    """Best-effort parameter count across sklearn and hand-rolled models."""
    explicit = getattr(model, "_param_count", None)
    if explicit is not None:
        return int(explicit)

    estimator = getattr(model, "estimator", None)
    for candidate in (model, estimator):
        if candidate is None:
            continue
        total = sum(int(np.asarray(getattr(candidate, attr)).size)
                    for attr in ("coef_", "intercept_") if getattr(candidate, attr, None) is not None)
        if total:
            return total
        estimators = getattr(candidate, "estimators_", None)
        if estimators is not None:
            try:
                return int(sum(int(e.tree_.node_count) * 2 for e in np.ravel(estimators)))
            except Exception:
                return int(np.size(estimators))
        predictors = getattr(candidate, "_predictors", None)  # HistGradientBoosting
        if predictors:
            try:
                return int(sum(int(p.get_n_leaf_nodes()) * 2 for stage in predictors for p in np.ravel(stage)))
            except Exception:
                return int(sum(len(np.ravel(stage)) for stage in predictors))
    return 0


class PrevalencePrior:
    """Predicts the training-set deterioration rate for every window. The floor."""

    _param_count = 1

    def fit(self, x, y, subject_ids=None):
        self.rate = float(np.mean(y == 1)) if len(y) else 0.5
        return self

    def predict_proba(self, x):
        return np.full(x.shape[0], self.rate, dtype=np.float32)

    @property
    def n_parameters(self):
        return 1


class SklearnGaitClassifier:
    """Wraps any sklearn estimator over window_features, with a train-fitted scaler."""

    def __init__(self, estimator: Any, sampling_rate_hz: float = 50.0) -> None:
        self.estimator = estimator
        self.sampling_rate_hz = sampling_rate_hz
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    def _prepare(self, x, fit_scaler=False):
        feats = window_features(x, self.sampling_rate_hz)
        if fit_scaler:
            self._mu = feats.mean(axis=0)
            self._sigma = feats.std(axis=0) + 1e-8
        return (feats - self._mu) / self._sigma if self._mu is not None else feats

    def fit(self, x, y, subject_ids=None):
        self.estimator.fit(self._prepare(x, fit_scaler=True), y)
        return self

    def predict_proba(self, x):
        feats = self._prepare(x)
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(feats)
            return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
        scores = self.estimator.decision_function(feats)
        return 1.0 / (1.0 + np.exp(-scores))

    @property
    def n_parameters(self):
        return count_parameters(self)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


class NumpyMLPMonitor:
    """Two-layer MLP over gait descriptors, trained with Adam in pure numpy.

    No torch dependency, so this is the one deep row that runs anywhere. It
    sees engineered features, not the raw waveform, so it isolates "does a
    non-linear boundary help" from "does representation learning help".
    """

    def __init__(self, hidden=64, epochs=200, learning_rate=1e-3, weight_decay=1e-4,
                 batch_size=64, sampling_rate_hz=50.0, seed=2026):
        self.hidden = hidden
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.sampling_rate_hz = sampling_rate_hz
        self.seed = seed
        self._weights: list[np.ndarray] = []
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    @property
    def n_parameters(self):
        return int(sum(w.size for w in self._weights))

    def _prepare(self, x, fit_scaler=False):
        feats = window_features(x, self.sampling_rate_hz)
        if fit_scaler:
            self._mu = feats.mean(axis=0)
            self._sigma = feats.std(axis=0) + 1e-8
        return (feats - self._mu) / self._sigma if self._mu is not None else feats

    @staticmethod
    def _forward(x, weights):
        w1, b1, w2, b2 = weights
        hidden = np.maximum(0.0, x @ w1 + b1)
        return hidden, (hidden @ w2 + b2).ravel()

    def fit(self, x, y, subject_ids=None):
        rng = np.random.default_rng(self.seed)
        features = self._prepare(x, fit_scaler=True)
        labels = np.asarray(y, dtype=float)
        n, n_features = features.shape
        if n < 2:
            raise ValueError(f"Only {n} training sample(s) -- cannot fit an MLP. Use gait_logreg instead.")

        w1 = rng.normal(0, np.sqrt(2.0 / n_features), (n_features, self.hidden))
        b1 = np.zeros(self.hidden)
        w2 = rng.normal(0, np.sqrt(1.0 / self.hidden), (self.hidden, 1))
        b2 = np.zeros(1)
        self._weights = [w1, b1, w2, b2]

        positive_rate = float(np.mean(labels == 1)) or 0.5
        pos_weight = (1 - positive_rate) / positive_rate

        moment1 = [np.zeros_like(w) for w in self._weights]
        moment2 = [np.zeros_like(w) for w in self._weights]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        batch_size = min(self.batch_size, n)

        for _ in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                index = order[start:start + batch_size]
                batch_x, batch_y = features[index], labels[index]
                m = batch_x.shape[0]

                hidden, logits = self._forward(batch_x, self._weights)
                probability = _sigmoid(logits)
                weights_per_sample = np.where(batch_y == 1, pos_weight, 1.0)
                d_logits = weights_per_sample * (probability - batch_y) / m

                w1, b1, w2, b2 = self._weights
                grad_w2 = hidden.T @ d_logits[:, None]
                grad_b2 = np.array([d_logits.sum()])
                d_hidden = np.outer(d_logits, w2.ravel()) * (hidden > 0)
                grad_w1 = batch_x.T @ d_hidden
                grad_b1 = d_hidden.sum(axis=0)
                gradients = [
                    grad_w1 + self.weight_decay * w1, grad_b1,
                    grad_w2 + self.weight_decay * w2, grad_b2,
                ]

                step += 1
                for i, gradient in enumerate(gradients):
                    moment1[i] = beta1 * moment1[i] + (1 - beta1) * gradient
                    moment2[i] = beta2 * moment2[i] + (1 - beta2) * gradient ** 2
                    corrected1 = moment1[i] / (1 - beta1 ** step)
                    corrected2 = moment2[i] / (1 - beta2 ** step)
                    self._weights[i] = self._weights[i] - self.learning_rate * corrected1 / (np.sqrt(corrected2) + eps)
        return self

    def predict_proba(self, x):
        _, logits = self._forward(self._prepare(x), self._weights)
        return _sigmoid(logits).astype(np.float32)


def build_model(key: str, sampling_rate_hz: float = 50.0, seed: int = 2026):
    if key == "prevalence_prior":
        return PrevalencePrior()
    if key == "gait_logreg":
        from sklearn.linear_model import LogisticRegression
        return SklearnGaitClassifier(LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"), sampling_rate_hz)
    if key == "gait_random_forest":
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
        return SklearnGaitClassifier(estimator, sampling_rate_hz)
    if key == "gait_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        estimator = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
        return SklearnGaitClassifier(estimator, sampling_rate_hz)
    if key == "descriptor_mlp":
        return NumpyMLPMonitor(sampling_rate_hz=sampling_rate_hz, seed=seed)
    raise KeyError(f"Unknown model '{key}'. Known: {MODEL_KEYS}")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def compute_metrics(y_true, y_score, threshold=0.5, subject_ids=None) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    y_pred = (y_score >= threshold).astype(int)

    out: dict[str, Any] = {"n_samples": int(y_true.size), "threshold": float(threshold)}
    out.update(_confusion_metrics(y_true, y_pred))
    out.update(_ranking_metrics(y_true, y_score))

    if subject_ids is not None:
        out["per_subject"] = _per_subject(y_true, y_score, np.asarray(subject_ids), threshold)
    return out


def _confusion_metrics(y_true, y_pred) -> dict[str, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = tp + tn + fp + fn

    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and not np.isnan(precision + recall) else 0.0)
    balanced = (np.nanmean([recall, specificity])
                if not (np.isnan(recall) and np.isnan(specificity)) else float("nan"))

    return {
        "accuracy": (tp + tn) / total if total else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced),
    }


def _ranking_metrics(y_true, y_score) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "auprc": float("nan")}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        return {"roc_auc": float(roc_auc_score(y_true, y_score)), "auprc": float(average_precision_score(y_true, y_score))}
    except ImportError:
        order = np.argsort(y_score)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(y_score) + 1)
        n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
        auc = (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        return {"roc_auc": float(auc), "auprc": float("nan")}


def _per_subject(y_true, y_score, subject_ids, threshold) -> dict[str, dict[str, float]]:
    result = {}
    for subject in np.unique(subject_ids):
        mask = subject_ids == subject
        scores = _confusion_metrics(y_true[mask], (y_score[mask] >= threshold).astype(int))
        result[str(subject)] = {"n_samples": int(mask.sum()), "balanced_accuracy": scores["balanced_accuracy"]}
    return result


# --------------------------------------------------------------------------
# Evaluation loop
# --------------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def evaluate(
    model_key: str,
    dataset_key: str,
    seed: int = 2026,
    sampling_rate_hz: float = 50.0,
    subject_wise_split: bool = True,
    data_root: str | Path = "data/raw",
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    """Fit one model on one dataset's train split, score it on test. Raises
    FileNotFoundError untouched -- the caller reports missing data, not a crash."""
    _seed_everything(seed)
    dataset_kwargs = {"sampling_rate_hz": sampling_rate_hz} if dataset_key == "synthetic_crew_gait" else {}

    train_set = datasets.load(dataset_key, split="train", seed=seed, data_root=data_root,
                               subject_wise_split=subject_wise_split, **dataset_kwargs)
    test_set = datasets.load(dataset_key, split="test", seed=seed, data_root=data_root,
                              subject_wise_split=subject_wise_split, **dataset_kwargs)

    if len(train_set) == 0 or len(test_set) == 0:
        return {"model": model_key, "dataset": dataset_key, "status": "empty_split",
                "error": f"train={len(train_set)} test={len(test_set)} windows"}

    model = build_model(model_key, sampling_rate_hz=sampling_rate_hz, seed=seed)
    model.fit(train_set.x, train_set.y, train_set.subject_ids)
    scores = np.asarray(model.predict_proba(test_set.x), dtype=float).ravel()

    metrics = compute_metrics(test_set.y, scores, threshold=decision_threshold,
                               subject_ids=test_set.subject_ids if subject_wise_split else None)

    row = {
        "model": model_key,
        "dataset": dataset_key,
        "status": "completed",
        "seed": seed,
        "n_train": len(train_set),
        "n_test": len(test_set),
        "n_subjects_test": int(np.unique(test_set.subject_ids).size),
        "n_parameters": model.n_parameters,
        "label_source": test_set.label_source,
        PRIMARY_METRIC: round(metrics[PRIMARY_METRIC], 4),
        "roc_auc": round(metrics["roc_auc"], 4) if not np.isnan(metrics["roc_auc"]) else float("nan"),
        "auprc": round(metrics["auprc"], 4) if not np.isnan(metrics["auprc"]) else float("nan"),
        "f1": round(metrics["f1"], 4),
    }
    if "per_subject" in metrics:
        # A pooled score over a handful of astronauts hides individual variance --
        # keep the per-subject breakdown, not just the aggregate.
        row["per_subject"] = json.dumps(metrics["per_subject"])
    row["magnitude_control_auroc"] = (
        round(test_set.magnitude_control_auroc, 4)
        if test_set.magnitude_control_auroc == test_set.magnitude_control_auroc
        else float("nan")
    )
    if test_set.label_source.startswith("derived"):
        row["validity_warning"] = "circular label -- plumbing check only, do not report"
    elif test_set.magnitude_control_auroc > 0.9:
        # A real label can still be trivially separable: preflight samples help
        # define their own baseline, so their delta_z is near zero by
        # arithmetic. Every model then scores ~1.000 for no good reason.
        row["validity_warning"] = (
            f"leaky representation -- mean |delta_z| alone scores AUROC "
            f"{test_set.magnitude_control_auroc:.3f}; see vlm_bench/reference_baselines.py"
        )
    return row


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _print_table(rows: list[dict[str, Any]]) -> None:
    cols = ["model", "dataset", "status", PRIMARY_METRIC, "roc_auc", "f1", "n_parameters", "seconds"]
    widths = {c: max([len(c)] + [len(str(r.get(c, ""))) for r in rows]) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("-" * (sum(widths.values()) + 2 * (len(cols) - 1)))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    for r in rows:
        if r.get("error"):
            print(f"\n  {r['model']}: {r['status']} -- {r['error']}")
        if r.get("validity_warning"):
            print(f"\n  {r['model']}: WARNING -- {r['validity_warning']}")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="synthetic_crew_gait",
                         help=f"'synthetic_crew_gait' or one of {list(datasets.REAL_DATASET_KEYS)}")
    parser.add_argument("--models", default="all", help=f"comma-separated subset of {MODEL_KEYS}, or 'all'")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sampling-rate-hz", type=float, default=50.0)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--no-subject-wise-split", dest="subject_wise_split", action="store_false", default=True,
                         help="Allow a crew member in more than one split. Inflates results -- see README.")
    parser.add_argument("--results-csv", default="results/results.csv")
    args = parser.parse_args(argv)

    model_keys = list(MODEL_KEYS) if args.models == "all" else [m.strip() for m in args.models.split(",")]

    if args.dataset == "synthetic_crew_gait":
        print("NOTE: synthetic_crew_gait is a smoke test -- never report these numbers.\n")

    rows = []
    for model_key in model_keys:
        started = time.perf_counter()
        try:
            row = evaluate(
                model_key, args.dataset, seed=args.seed, sampling_rate_hz=args.sampling_rate_hz,
                subject_wise_split=args.subject_wise_split, data_root=args.data_root,
                decision_threshold=args.decision_threshold,
            )
        except FileNotFoundError as exc:
            row = {"model": model_key, "dataset": args.dataset, "status": "missing_data", "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 -- one bad model must not kill the run
            row = {"model": model_key, "dataset": args.dataset, "status": "failed",
                   "error": f"{type(exc).__name__}: {exc}"}
        row["seconds"] = round(time.perf_counter() - started, 2)
        rows.append(row)

    _print_table(rows)
    results_path = Path(args.results_csv)
    _write_csv(rows, results_path)
    print(f"\nwrote {results_path}")

    return 0 if all(r.get("status") == "completed" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
