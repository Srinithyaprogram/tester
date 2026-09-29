"""
Crew deterioration-detection datasets.

Two shapes of data:

* ``synthetic_crew_gait`` -- a simulated 6-channel IMU corpus, generated on the
  fly. Deliberately imperfect (lands around 0.80-0.88 balanced accuracy) so
  pipeline breakage is visible. It is a smoke test -- never report results
  from it.

* everything else -- a tidy longitudinal panel CSV at
  ``data/raw/<dataset_key>/*.csv``, one row per crew member per mission
  timepoint (``subject_id``, ``timepoint``, optional ``phase``/``label``, and
  any number of numeric measures). See docs/DATASETS.md for the exact shape,
  and ``scripts/make_example_panel.py`` / ``scripts/validate_panel.py`` for
  generating and checking one.

Known real dataset keys (registered corpora, no loader special-casing needed
beyond "read the panel"): osdr_human_spaceflight, genelab_omics,
alsda_physiology, osdr_rodent_behaviour, inspiration4, genelab_astronaut,
osdr_bedrest. See docs/OSDR_FINDINGS.md before treating any of these as a
movement dataset -- human movement/gait/posture data does not exist in OSDR.

Three of these keys are backed by real, downloadable data today:
``inspiration4`` (clinical labs, 4 crew), ``genelab_astronaut`` (OSD-174, 10
ISS astronauts) and ``osdr_bedrest`` (OSD-370, 10 bed-rest subjects). The
latter two ship a real ``label`` column taken from NASA's own deposited factor
values, so they do not trigger the circular baseline-deviation fallback below.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REAL_DATASET_KEYS = (
    "osdr_human_spaceflight",
    "genelab_omics",
    "alsda_physiology",
    "osdr_rodent_behaviour",
    "inspiration4",
    "genelab_astronaut",
    "osdr_bedrest",
)


@dataclass
class CrewWindowSet:
    """One split of crew samples, ready to hand to a model."""

    x: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    split: str = "test"
    feature_names: tuple[str, ...] = ()
    #: "provided" when the corpus ships labels; "derived_*" when we computed
    #: them from the same features the model receives (circular -- see below).
    label_source: str = "provided"
    #: AUROC of the mean absolute baseline-deviation z-score alone -- a single
    #: scalar, no model. See the note in statistics().
    magnitude_control_auroc: float = float("nan")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def statistics(self) -> dict:
        unique, counts = np.unique(self.y, return_counts=True)
        stats = {
            "split": self.split,
            "n_samples": len(self),
            "n_subjects": int(np.unique(self.subject_ids).size),
            "class_counts": {str(int(u)): int(c) for u, c in zip(unique, counts)},
            "positive_rate": float(np.mean(self.y == 1)) if len(self) else 0.0,
            "label_source": self.label_source,
        }
        if self.label_source.startswith("derived"):
            stats["validity_warning"] = (
                "CIRCULAR LABEL: derived from the same baseline-deviation "
                "features the model receives as input. Plumbing check only."
            )
        # A provided label can still be trivially separable, for a reason that
        # has nothing to do with physiology: preflight samples are among the
        # samples used to compute their own baseline, so their delta_z is
        # pinned near zero by arithmetic while exposed samples are unbounded.
        # When that alone separates the classes, every model here scores ~1.000
        # and none of it means anything.
        stats["magnitude_control_auroc"] = self.magnitude_control_auroc
        if self.magnitude_control_auroc == self.magnitude_control_auroc and \
                self.magnitude_control_auroc > 0.9:
            stats["validity_warning"] = (
                f"LEAKY REPRESENTATION: mean |delta_z| alone separates the "
                f"classes at AUROC {self.magnitude_control_auroc:.3f} -- no model "
                f"needed. Scores here measure deviation magnitude, not "
                f"physiology. Use vlm_bench/reference_baselines.py, which "
                f"z-scores cohort-wide and cross-validates leave-one-subject-out."
            )
        return stats


def load(
    dataset_key: str,
    split: str = "test",
    seed: int = 2026,
    data_root: str | Path = "data/raw",
    subject_wise_split: bool = True,
    **kwargs,
) -> CrewWindowSet:
    """Load one split of ``dataset_key``.

    ``synthetic_crew_gait`` is generated in memory. Every other key resolves
    to a CSV/TSV under ``data_root/<dataset_key>/``.
    """
    if dataset_key == "synthetic_crew_gait":
        return _load_synthetic(split=split, seed=seed, subject_wise_split=subject_wise_split, **kwargs)
    return _load_longitudinal_panel(
        dataset_key, split=split, seed=seed, data_root=Path(data_root),
        subject_wise_split=subject_wise_split, **kwargs,
    )


# --------------------------------------------------------------------------
# Synthetic smoke-test corpus
# --------------------------------------------------------------------------


def _load_synthetic(
    split: str,
    seed: int,
    subject_wise_split: bool,
    n_subjects: int = 24,
    n_samples: int = 2400,
    window_seconds: float = 10.0,
    sampling_rate_hz: float = 50.0,
) -> CrewWindowSet:
    """Simulate crew IMU windows with a physiologically-motivated label rule.

    Deterioration is injected as reduced gait cadence, amplified
    medio-lateral sway, and elevated stride-time variability -- the three
    effects most consistently reported in microgravity deconditioning and
    bed-rest studies. Effect sizes are deliberately small and overlapping so a
    suspiciously perfect score reads as a bug, not a result.
    """
    rng = np.random.default_rng(seed)
    window_length = int(round(window_seconds * sampling_rate_hz))
    per_subject = max(1, n_samples // n_subjects)
    channels = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
    t = np.arange(window_length) / sampling_rate_hz

    xs, ys, sids = [], [], []
    for subject in range(n_subjects):
        base_cadence = rng.normal(1.9, 0.12)
        base_sway = rng.normal(0.08, 0.018)
        base_noise = rng.uniform(0.06, 0.20)
        deteriorated = rng.random(per_subject) < 0.35

        for is_bad in deteriorated:
            severity = rng.uniform(0.35, 1.0) if is_bad else 0.0
            cadence = base_cadence * (1.0 - 0.16 * severity)
            sway = base_sway * (1.0 + 1.10 * severity)
            stride_jitter = 0.03 + 0.07 * severity

            phase = 2 * np.pi * cadence * t + rng.normal(0, stride_jitter, window_length).cumsum() * 0.02
            noise = base_noise
            acc_x = np.sin(phase) + rng.normal(0, noise, window_length)
            acc_y = sway * np.sin(0.5 * phase + 0.7) + rng.normal(0, noise, window_length)
            acc_z = 0.6 * np.cos(phase) + 1.0 + rng.normal(0, noise, window_length)
            gyro_x = np.gradient(acc_x) * sampling_rate_hz * 0.1
            gyro_y = np.gradient(acc_y) * sampling_rate_hz * 0.1
            gyro_z = np.gradient(acc_z) * sampling_rate_hz * 0.1

            xs.append(np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]))
            ys.append(int(is_bad))
            sids.append(subject)

    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    subject_ids = np.asarray(sids, dtype=np.int64)

    x, y, subject_ids = _apply_split(x, y, subject_ids, split, subject_wise_split, seed=seed)
    return CrewWindowSet(x=x, y=y, subject_ids=subject_ids, split=split, feature_names=channels)


# --------------------------------------------------------------------------
# Real corpora: tidy longitudinal panel CSV
# --------------------------------------------------------------------------


def _load_longitudinal_panel(
    dataset_key: str,
    split: str,
    seed: int,
    data_root: Path,
    subject_wise_split: bool,
    deterioration_z: float = 2.0,
) -> CrewWindowSet:
    """Load a tidy longitudinal crew panel (OSDR / GeneLab / ALSDA / I4 shape).

    A "sample" is one crew member at one mission timepoint. Every sample is
    emitted as two stacked channels: the cohort-standardised value and its
    delta from that crew member's own preflight baseline. See docs/DATASETS.md.
    """
    root = data_root / dataset_key
    candidates = sorted(list(root.glob("*.csv")) + list(root.glob("*.tsv"))) if root.is_dir() else [root]
    if not candidates or not candidates[0].exists():
        raise FileNotFoundError(
            f"No panel for '{dataset_key}' under {root}. Export a tidy panel.csv "
            f"there -- see docs/DATASETS.md -- then re-run. "
            f"(python scripts/make_example_panel.py --dataset {dataset_key} shows the format.)"
        )

    path = candidates[0]
    frame = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    frame.columns = [str(c).strip() for c in frame.columns]

    missing = {"subject_id", "timepoint"} - set(frame.columns)
    if missing:
        raise ValueError(f"'{path.name}' is missing required column(s): {sorted(missing)}")

    if "phase" not in frame.columns:
        frame["phase"] = frame["timepoint"].astype(str).map(_infer_phase)

    # 'label_acute' and 'phase_source' are alternative encodings of the target
    # and of 'phase'. They are numeric/stringly-typed like a measurement, so
    # they must be named here or they leak the label into the feature matrix.
    reserved = {
        "subject_id", "timepoint", "phase", "phase_source", "label", "label_acute",
        "split", "sample_id", "accession",
    }
    feature_cols = [c for c in frame.columns if c not in reserved and pd.api.types.is_numeric_dtype(frame[c])]
    if not feature_cols:
        raise ValueError(f"'{path.name}' contains no numeric feature columns.")

    frame = frame.sort_values(["subject_id", "timepoint"]).reset_index(drop=True)
    values = frame[feature_cols].astype(float)
    values = values.fillna(values.median(numeric_only=True)).fillna(0.0)

    preflight = frame["phase"].astype(str).str.lower().eq("preflight")
    grouped = values.groupby(frame["subject_id"])
    baseline_mean = values[preflight].groupby(frame.loc[preflight, "subject_id"]).mean()
    baseline_std = values[preflight].groupby(frame.loc[preflight, "subject_id"]).std()
    baseline_mean = baseline_mean.reindex(grouped.mean().index).fillna(grouped.mean())
    baseline_std = baseline_std.reindex(grouped.std().index).fillna(grouped.std()).fillna(1.0)
    baseline_std = baseline_std.replace(0.0, 1.0)

    aligned_mean = baseline_mean.loc[frame["subject_id"]].to_numpy(dtype=np.float32)
    aligned_std = baseline_std.loc[frame["subject_id"]].to_numpy(dtype=np.float32)
    raw = values.to_numpy(dtype=np.float32)
    delta_z = (raw - aligned_mean) / (aligned_std + 1e-8)
    raw_z = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-8)
    x = np.stack([raw_z, delta_z], axis=1).astype(np.float32)  # (n, 2, n_features)

    if "label" in frame.columns:
        y = frame["label"].to_numpy(dtype=np.int64)
        label_source = "provided"
    else:
        severity = np.sqrt(np.mean(delta_z ** 2, axis=1))
        y = (severity >= deterioration_z).astype(np.int64)
        label_source = f"derived_baseline_deviation_z{deterioration_z:g}"
        warnings.warn(
            f"'{dataset_key}' has no 'label' column, so labels were derived from "
            f"the same baseline-deviation features the model sees as input. This "
            f"is CIRCULAR -- use it to check plumbing only. Add a clinical "
            f"'label' column before reporting anything from this corpus.",
            UserWarning, stacklevel=2,
        )

    subject_codes = frame["subject_id"].astype("category").cat.codes.to_numpy(dtype=np.int64)
    # Measured on the full corpus, before splitting -- a per-split estimate on
    # two held-out subjects would be too noisy to act on.
    magnitude_auroc = _magnitude_control_auroc(delta_z, y)

    if "split" in frame.columns:
        mask = frame["split"].astype(str).to_numpy() == split
        x, y, subject_codes = x[mask], y[mask], subject_codes[mask]
    else:
        x, y, subject_codes = _apply_split(x, y, subject_codes, split, subject_wise_split, seed=seed)

    return CrewWindowSet(
        x=x, y=y, subject_ids=subject_codes, split=split,
        feature_names=tuple(feature_cols), label_source=label_source,
        magnitude_control_auroc=magnitude_auroc,
    )


def _magnitude_control_auroc(delta_z: np.ndarray, y: np.ndarray) -> float:
    """AUROC of 'how far from baseline is this sample', as one number.

    torch-free and sklearn-free by design -- this module has no heavy deps and
    the trapezoid over a sorted score is four lines.
    """
    if len(np.unique(y)) < 2:
        return float("nan")
    score = np.abs(delta_z).mean(axis=1)
    order = np.argsort(score)
    labels = y[order]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    # Mann-Whitney U over ranks, ties broken by position (adequate for a guard).
    rank_sum = float(np.flatnonzero(labels == 1).sum() + n_pos)
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _infer_phase(timepoint: str) -> str:
    label = str(timepoint).strip().upper()
    if label.startswith("L-") or label.startswith("PRE"):
        return "preflight"
    if label.startswith("FD") or label.startswith("IN") or label.startswith("MD"):
        return "inflight"
    if label.startswith("R+") or label.startswith("POST"):
        return "postflight"
    return "unknown"


def _apply_split(
    x: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    split: str,
    subject_wise: bool,
    seed: int,
    ratios: tuple[float, float, float] = (0.6, 0.15, 0.25),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic train/val/test partition.

    ``subject_wise=True`` (the default, and the only defensible choice for a
    crew of ~20) assigns whole astronauts to a split, so no window from a
    given crew member leaks between train and test.
    """
    rng = np.random.default_rng(seed)

    if subject_wise:
        subjects = np.unique(subject_ids)
        rng.shuffle(subjects)
        n_train = int(round(ratios[0] * len(subjects)))
        n_val = int(round(ratios[1] * len(subjects)))
        buckets = {
            "train": set(subjects[:n_train].tolist()),
            "val": set(subjects[n_train:n_train + n_val].tolist()),
            "test": set(subjects[n_train + n_val:].tolist()),
        }
        mask = np.isin(subject_ids, list(buckets[split]))
    else:
        idx = np.arange(len(y))
        rng.shuffle(idx)
        n_train = int(round(ratios[0] * len(idx)))
        n_val = int(round(ratios[1] * len(idx)))
        buckets = {
            "train": idx[:n_train],
            "val": idx[n_train:n_train + n_val],
            "test": idx[n_train + n_val:],
        }
        mask = np.zeros(len(y), dtype=bool)
        mask[buckets[split]] = True

    return x[mask], y[mask], subject_ids[mask]
