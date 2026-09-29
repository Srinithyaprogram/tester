#!/usr/bin/env python3
"""
Confidence intervals and significance tests for Tables 1-4.

Without these the tables invite over-reading. Inspiration4 has four crew
members; GeneLab and bed rest have ten subjects each. A gap of 0.05 balanced
accuracy between two models on n=28 samples from 4 people is not evidence of
anything, and nothing in the point estimates says so.

    python vlm_bench/astro_stats.py ci        # per model x corpus
    python vlm_bench/astro_stats.py compare   # the comparisons the paper makes
    python vlm_bench/astro_stats.py all

Resampling is at the **subject** level, not the sample level. Samples from one
crew member are repeated measures of the same person and are not independent;
bootstrapping over samples would produce intervals several times too narrow.
Resampling whole subjects is the analogue of the leave-one-subject-out protocol
the point estimates already use.

Model-vs-model tests are **paired** on the resampled subject set: both models
score the same bootstrap cohort, and the statistic is the difference. Pairing
removes cohort difficulty as a source of variance, which matters enormously
when there are only four crew members to resample.

The reported p-value is two-sided, from the bootstrap distribution of the
difference: 2 * min(P(d <= 0), P(d >= 0)). With 4 subjects the smallest
attainable p-value is bounded well away from 0 -- that is a property of the
data, not a defect of the test, and it is the point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from astro_eval import (  # noqa: E402
    DATASET_PRETTY, DATASETS, PRETTY, linear_probe_loso, load_encoding,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent / "results"

MODELS = ("clip", "blip2", "llava16", "qwen25vl", "biomedclip", "medflamingo",
          "biovilt", "chexagent")
FAMILY = {
    "clip": "general-purpose", "blip2": "general-purpose",
    "llava16": "general-purpose", "qwen25vl": "general-purpose",
    "biomedclip": "biomedical", "medflamingo": "biomedical",
    "biovilt": "biomedical", "chexagent": "biomedical",
}

N_BOOT = 2000
SEED = 2026


def _balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    """Mean of per-class recall. Returns NaN if a class is absent."""
    out = []
    for cls in (0, 1):
        mask = y == cls
        if not mask.any():
            return float("nan")
        out.append(float((pred[mask] == cls).mean()))
    return float(np.mean(out))


def _mcc(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float(((y == 1) & (pred == 1)).sum())
    tn = float(((y == 0) & (pred == 0)).sum())
    fp = float(((y == 0) & (pred == 1)).sum())
    fn = float(((y == 1) & (pred == 0)).sum())
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denominator) if denominator > 0 else 0.0


METRIC_FNS = {"balanced_accuracy": _balanced_accuracy, "mcc": _mcc}


def subject_bootstrap_indices(subjects: np.ndarray, rng: np.random.Generator):
    """Yield sample indices for one bootstrap cohort of resampled subjects.

    A subject drawn twice contributes its samples twice -- that is what makes
    the interval reflect between-subject variability.
    """
    unique = np.unique(subjects)
    drawn = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(subjects == s) for s in drawn])


def predictions(dataset: str, model: str, protocol: str, seed: int = SEED):
    """(y, pred, subjects) for one model x corpus under one protocol."""
    data = load_encoding(dataset, model)
    if data is None:
        return None
    y = data["labels"].astype(int)
    subjects = data["subjects"]
    if protocol == "zero_shot":
        pred = (data["zeroshot_score"] >= 0.5).astype(int)
    else:
        pred, _ = linear_probe_loso(data["image_embed"], y, subjects, seed)
    return y, pred, subjects


def bootstrap_ci(y, pred, subjects, metric: str, n_boot: int = N_BOOT,
                 seed: int = SEED) -> tuple[float, float, float]:
    """Point estimate and percentile 95% CI over resampled subjects."""
    fn = METRIC_FNS[metric]
    point = fn(y, pred)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = subject_bootstrap_indices(subjects, rng)
        value = fn(y[idx], pred[idx])
        if value == value:  # skip cohorts that lost a class entirely
            draws.append(value)
    if not draws:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_test(y, pred_a, pred_b, subjects, metric: str,
                n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Paired subject bootstrap of metric(a) - metric(b)."""
    fn = METRIC_FNS[metric]
    observed = fn(y, pred_a) - fn(y, pred_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = subject_bootstrap_indices(subjects, rng)
        d = fn(y[idx], pred_a[idx]) - fn(y[idx], pred_b[idx])
        if d == d:
            diffs.append(d)
    if not diffs:
        return {"difference": observed, "ci_low": float("nan"),
                "ci_high": float("nan"), "p_value": float("nan")}
    diffs = np.asarray(diffs)
    p = 2 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean()))
    return {
        "difference": observed,
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_value": min(1.0, p),
    }


def build_ci_table(n_boot: int) -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for protocol in ("zero_shot", "linear_probe_loso"):
                got = predictions(dataset, model, protocol)
                if got is None:
                    continue
                y, pred, subjects = got
                row = {
                    "model": PRETTY.get(model, model),
                    "family": FAMILY[model],
                    "dataset": DATASET_PRETTY.get(dataset, dataset),
                    "protocol": protocol,
                    "n_subjects": int(len(np.unique(subjects))),
                }
                for metric in METRIC_FNS:
                    point, low, high = bootstrap_ci(y, pred, subjects, metric, n_boot)
                    row[metric] = point
                    row[f"{metric}_ci_low"] = low
                    row[f"{metric}_ci_high"] = high
                rows.append(row)
    return pd.DataFrame(rows)


def build_comparison_table(n_boot: int) -> pd.DataFrame:
    """The three comparisons the write-up actually leans on."""
    rows = []

    for dataset in DATASETS:
        available = {m: predictions(dataset, m, "linear_probe_loso") for m in MODELS}
        available = {m: v for m, v in available.items() if v is not None}
        if not available:
            continue
        y = next(iter(available.values()))[0]
        subjects = next(iter(available.values()))[2]

        # RQ2: best general-purpose vs best biomedical, chosen on the point
        # estimate then tested -- the same (slightly optimistic) selection the
        # narrative makes, tested honestly.
        def best(family):
            pool = {m: v for m, v in available.items() if FAMILY[m] == family}
            return max(pool, key=lambda m: _balanced_accuracy(y, pool[m][1])) if pool else None

        gen, bio = best("general-purpose"), best("biomedical")
        if gen and bio:
            for metric in METRIC_FNS:
                result = paired_test(
                    y, available[gen][1], available[bio][1], subjects, metric, n_boot
                )
                rows.append({
                    "comparison": "RQ2: best general-purpose - best biomedical",
                    "detail": f"{PRETTY[gen]} vs {PRETTY[bio]}",
                    "dataset": DATASET_PRETTY[dataset],
                    "metric": metric, **result,
                })

        # Zero-shot vs linear probe, per model.
        for model in available:
            zs = predictions(dataset, model, "zero_shot")
            if zs is None:
                continue
            result = paired_test(
                y, available[model][1], zs[1], subjects, "balanced_accuracy", n_boot
            )
            rows.append({
                "comparison": "linear probe - zero shot",
                "detail": PRETTY[model],
                "dataset": DATASET_PRETTY[dataset],
                "metric": "balanced_accuracy", **result,
            })

        # Best VLM probe vs the strongest raw-feature baseline.
        ceiling = _raw_feature_predictions(dataset)
        if ceiling is not None:
            ceiling_pred, ceiling_name = ceiling
            best_overall = max(available, key=lambda m: _balanced_accuracy(y, available[m][1]))
            result = paired_test(
                y, available[best_overall][1], ceiling_pred, subjects,
                "balanced_accuracy", n_boot,
            )
            rows.append({
                "comparison": "best VLM probe - raw-feature ceiling",
                "detail": f"{PRETTY[best_overall]} vs {ceiling_name}",
                "dataset": DATASET_PRETTY[dataset],
                "metric": "balanced_accuracy", **result,
            })
    return pd.DataFrame(rows)


def _raw_feature_predictions(dataset: str) -> tuple[np.ndarray, str] | None:
    """Leave-one-subject-out predictions from the raw measurements.

    Returns the *strongest* of the raw-feature baselines, not just logistic
    regression: on GeneLab the gradient-boosted tree is far better (0.738 vs
    0.600 balanced accuracy), and testing a VLM against the weaker of the two
    would manufacture a "VLM beats the raw features" result out of a poorly
    chosen reference.

    Aligned to the encoding's sample order so the paired test compares the same
    samples -- reference_baselines.py reads the panel directly and its row order
    need not match the manifest's.
    """
    from reference_baselines import load_panel

    from sklearn.base import clone
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = load_encoding(dataset, "clip")
    if data is None:
        return None
    try:
        x, y, groups, _ = load_panel(dataset)
    except FileNotFoundError:
        return None
    if len(y) != len(data["labels"]) or not np.array_equal(y, data["labels"]):
        return None

    # Raw panels carry NaN (Inspiration4 has a >50%-missing analyte), so this
    # needs the imputing pipeline from reference_baselines.py rather than
    # astro_eval's bare probe -- embeddings never have missing values.
    # Imputation happens inside the fold, so no test-fold value informs it.
    candidates = {
        "logistic regression": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(C=0.01, max_iter=5000, random_state=SEED),
        ),
        "gradient boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=200, max_depth=3, learning_rate=0.1, random_state=SEED
            ),
        ),
    }

    best_pred, best_name, best_score = None, "", -np.inf
    for name, model in candidates.items():
        pred = np.zeros(len(y), dtype=int)
        for train_idx, test_idx in LeaveOneGroupOut().split(x, y, groups):
            if len(np.unique(y[train_idx])) < 2:
                pred[test_idx] = int(y[train_idx][0])
                continue
            pred[test_idx] = clone(model).fit(x[train_idx], y[train_idx]).predict(x[test_idx])
        score = _balanced_accuracy(y, pred)
        if score > best_score:
            best_pred, best_name, best_score = pred, name, score
    return best_pred, best_name


def _report(frame: pd.DataFrame, name: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.csv"
    frame.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="--"))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("ci", "compare", "all"))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)

    if args.command in ("ci", "all"):
        print("=" * 100)
        print("95% CI over resampled subjects")
        print("=" * 100)
        _report(build_ci_table(args.n_boot), "stats_ci")
    if args.command in ("compare", "all"):
        print("\n" + "=" * 100)
        print("Paired subject bootstrap")
        print("=" * 100)
        _report(build_comparison_table(args.n_boot), "stats_comparisons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
