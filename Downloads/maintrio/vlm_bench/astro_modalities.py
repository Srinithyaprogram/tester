#!/usr/bin/env python3
"""
Render an astronaut health panel into the three modalities the benchmark needs.

The corpora are tabular (clinical labs, microarray intensities) but every model
in Tables 1-4 is a Vision-Language Model, so the table has to become something a
vision tower and a text tower can read:

    image    each sample's feature vector as a heatmap PNG, z-scored against
             the crew member's own preflight baseline
    text     a short clinical-style description naming the most-changed
             measurements, under CLIP's 77-token limit
    physio   the raw numeric vector, untouched -- the control arm for RQ4

All three are derived from the same row, so the RQ4 ablation compares
representations of identical information rather than different data.

    python vlm_bench/astro_modalities.py --dataset genelab_astronaut

Writes data/vlm_astro/<dataset>/{images/, manifest.json}.

## Why the default rendering is cohort_z, not baseline_z

The obvious rendering -- z-score each sample against that crew member's own
preflight draws -- leaks the label, and badly. A preflight sample is one of the
handful of samples used to *compute* that baseline, so its deviation is pinned
near zero by arithmetic, while a postflight sample has no such bound. The
result is that preflight heatmaps render nearly flat and postflight ones render
noisy, and the task collapses into "is this image colourful".

Measured, not assumed: under baseline_z a single scalar -- the mean absolute
z-score, no model involved -- separates the classes at AUROC 0.984 / 1.000 /
1.000 on the three corpora, and CLIP zero-shot scored 0.995 on Inspiration4.
Those are not detection results.

`cohort_z` z-scores each measurement across the whole corpus instead, so both
classes sit on the same footing and a model has to identify *which* analytes
moved rather than how much anything moved. The same magnitude control drops to
0.568 / 0.764 / 0.570. The residual is real physiology, and it is reported as a
row in vlm_bench/reference_baselines.py so every table can be read against it.

`baseline_z` is kept for inspection only and warns when selected.

The label itself is the flight phase from the mission timeline (see
scripts/build_osdr_panels.py) and is never a function of the measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
OUT_ROOT = REPO_ROOT / "data" / "vlm_astro"

DATASETS = ("inspiration4", "genelab_astronaut", "osdr_bedrest")

METADATA_COLUMNS = {
    "subject_id", "timepoint", "phase", "phase_source", "label", "label_acute",
    "split", "sample_id", "accession", "sex", "treatment",
}

#: Zero-shot prompt pair. Index 0 is the negative (label 0) class.
PROMPTS = {
    "inspiration4": (
        "a preflight blood biomarker panel from a healthy astronaut at baseline",
        "a postflight blood biomarker panel from an astronaut showing "
        "physiological stress after spaceflight",
    ),
    "genelab_astronaut": (
        "a preflight gene expression profile from a healthy astronaut at baseline",
        "a spaceflight gene expression profile from an astronaut showing "
        "physiological stress in orbit",
    ),
    "osdr_bedrest": (
        "a skeletal muscle gene expression profile from a healthy ambulatory "
        "subject at baseline",
        "a skeletal muscle gene expression profile from a subject deconditioned "
        "by prolonged bed rest",
    ),
}

#: How many measurements the text description is allowed to name. CLIP's text
#: encoder truncates at 77 tokens; five symbols plus framing fits.
TEXT_TOP_K = 5
#: Heatmap grid cap -- more cells than this and each is under a pixel.
MAX_IMAGE_FEATURES = 256


def load_panel(dataset: str, label_col: str = "label") -> tuple[pd.DataFrame, list[str]]:
    path = RAW_ROOT / dataset / "panel.csv"
    if not path.exists():
        raise FileNotFoundError(f"no panel for '{dataset}' at {path}")
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[frame.get(label_col, 0) >= 0].reset_index(drop=True)
    features = [
        c for c in frame.columns
        if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if not features:
        raise ValueError(f"'{dataset}' has no numeric feature columns")
    return frame, features


#: A crew member's baseline std comes from as few as two preflight draws, so it
#: can land near zero by chance and send that feature's z-score into the
#: thousands. Floor it at this fraction of the measurement's cohort-wide spread,
#: then clip: one unstable analyte must not dominate the heatmap or the probe.
BASELINE_STD_FLOOR = 0.25
Z_CLIP = 8.0


def cohort_z(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Z-score every measurement across the whole corpus. The default.

    Both classes are scaled identically, so nothing about a sample's magnitude
    is determined by which class it belongs to. See the module docstring for
    why the subject-baseline alternative is not usable as a default.
    """
    values = frame[features].astype(float)
    values = values.fillna(values.median(numeric_only=True)).fillna(0.0)
    raw = values.to_numpy(dtype=float)
    std = raw.std(axis=0, keepdims=True)
    z = (raw - raw.mean(axis=0, keepdims=True)) / np.where(std < 1e-8, 1.0, std)
    return np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -Z_CLIP, Z_CLIP)


def baseline_z(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Z-score every measurement against that subject's own preflight rows.

    Subjects with no preflight sample fall back to their own overall mean, so a
    cohort missing a baseline degrades instead of producing NaNs.

    LEAKY -- see the module docstring. Inspection only.
    """
    values = frame[features].astype(float)
    values = values.fillna(values.median(numeric_only=True)).fillna(0.0)
    preflight = frame["phase"].astype(str).str.lower().eq("preflight")

    overall_mean = values.groupby(frame["subject_id"]).transform("mean")
    overall_std = values.groupby(frame["subject_id"]).transform("std")
    if preflight.any():
        base_mean = values[preflight].groupby(frame.loc[preflight, "subject_id"]).mean()
        base_std = values[preflight].groupby(frame.loc[preflight, "subject_id"]).std()
        mean = base_mean.reindex(frame["subject_id"]).to_numpy(dtype=float)
        std = base_std.reindex(frame["subject_id"]).to_numpy(dtype=float)
        mean = np.where(np.isnan(mean), overall_mean.to_numpy(dtype=float), mean)
        std = np.where(np.isnan(std), overall_std.to_numpy(dtype=float), std)
    else:
        mean = overall_mean.to_numpy(dtype=float)
        std = overall_std.to_numpy(dtype=float)

    cohort_std = values.to_numpy(dtype=float).std(axis=0, keepdims=True)
    floor = np.where(cohort_std > 1e-8, BASELINE_STD_FLOOR * cohort_std, 1.0)
    std = np.where(~np.isfinite(std), floor, std)
    std = np.maximum(std, floor)

    z = (values.to_numpy(dtype=float) - mean) / std
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(z, -Z_CLIP, Z_CLIP)


def _pretty(name: str) -> str:
    """Turn a panel column name into something a text encoder can use."""
    if "|" in name:  # gene-symbol|probe from build_osdr_panels
        return name.split("|", 1)[0]
    tail = name.split("__", 1)[-1]
    for suffix in ("_value", "_concentration", "_percent"):
        tail = tail.replace(suffix, "")
    return tail.replace("_", " ").strip()


#: How the text should describe what the z-scores are relative to. Must track
#: the rendering, or the description asserts a comparison that was not made.
REFERENCE_WORDING = {
    "cohort_z": "relative to the cohort mean",
    "baseline_z": "relative to preflight baseline",
}


def describe(z_row: np.ndarray, features: list[str], dataset: str,
             rendering: str = "cohort_z", top_k: int = TEXT_TOP_K) -> str:
    """One short clinical-style sentence naming the most-changed measurements."""
    order = np.argsort(-np.abs(z_row))[:top_k]
    up = [_pretty(features[i]) for i in order if z_row[i] > 0]
    down = [_pretty(features[i]) for i in order if z_row[i] <= 0]
    subject = "gene expression" if dataset != "inspiration4" else "blood biomarker panel"

    parts = [f"Astronaut {subject} {REFERENCE_WORDING[rendering]}."]
    if up:
        parts.append("Increased: " + ", ".join(up[:3]) + ".")
    if down:
        parts.append("Decreased: " + ", ".join(down[:3]) + ".")
    n_large = int((np.abs(z_row) > 2).sum())
    parts.append(f"{n_large} measurements beyond two standard deviations.")
    return " ".join(parts)


RENDERINGS = {"cohort_z": cohort_z, "baseline_z": baseline_z}


def render(dataset: str, label_col: str = "label", overwrite: bool = False,
           rendering: str = "cohort_z") -> Path:
    """Write heatmap PNGs + a manifest carrying text, label and subject id."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame, features = load_panel(dataset, label_col)
    z = RENDERINGS[rendering](frame, features)

    # Keep the highest-variance measurements so the heatmap has visible
    # structure instead of a grid of near-identical cells.
    if z.shape[1] > MAX_IMAGE_FEATURES:
        keep = np.argsort(-z.var(axis=0))[:MAX_IMAGE_FEATURES]
        keep = np.sort(keep)
    else:
        keep = np.arange(z.shape[1])

    out_dir = OUT_ROOT / dataset
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    side = int(np.ceil(np.sqrt(len(keep))))
    samples: list[dict] = []
    for i in range(len(frame)):
        sample_id = f"{dataset}_{frame['subject_id'][i]}_{frame['timepoint'][i]}_{i}"
        sample_id = sample_id.replace("/", "-").replace(" ", "-")
        image_path = image_dir / f"{sample_id}.png"

        if overwrite or not image_path.exists():
            grid = np.full(side * side, np.nan)
            grid[: len(keep)] = z[i, keep]
            fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
            ax.imshow(grid.reshape(side, side), cmap="coolwarm", vmin=-3, vmax=3)
            ax.set_axis_off()
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(image_path, dpi=100)
            plt.close(fig)

        samples.append({
            "id": sample_id,
            "image_path": str(image_path.relative_to(REPO_ROOT)),
            "text": describe(z[i], features, dataset, rendering),
            "physio": [float(v) for v in z[i]],
            "dataset": dataset,
            "subject_id": str(frame["subject_id"][i]),
            "timepoint": str(frame["timepoint"][i]),
            "phase": str(frame["phase"][i]),
            "label": int(frame[label_col][i]),
        })

    magnitude_auroc = _magnitude_control(z, np.array([s["label"] for s in samples]))
    manifest = {
        "dataset": dataset,
        "label_col": label_col,
        "rendering": rendering,
        "magnitude_control_auroc": magnitude_auroc,
        "n_samples": len(samples),
        "n_features": int(z.shape[1]),
        "n_image_features": int(len(keep)),
        "prompts": list(PROMPTS[dataset]),
        "feature_names": features,
        "samples": samples,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    labels = np.array([s["label"] for s in samples])
    print(
        f"  {dataset}: {len(samples)} samples | {len(keep)} image features "
        f"| label 1/0 = {int(labels.sum())}/{int((labels == 0).sum())}"
    )
    print(f"    -> {manifest_path.relative_to(REPO_ROOT)}")
    print(f"    magnitude control (mean |z| alone, no model): AUROC {magnitude_auroc:.3f}")
    if magnitude_auroc > 0.9:
        print("    ** the rendering leaks the label -- do not report results on it **")
    print(f'    text sample: "{samples[0]["text"]}"')
    return manifest_path


def _magnitude_control(z: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of 'mean absolute z-score', the dumbest possible predictor.

    Any model that fails to beat this is reading overall deviation magnitude,
    not physiology. Recorded in the manifest so it travels with the results.
    """
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.abs(z).mean(axis=1)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, help="default: all")
    parser.add_argument("--label-col", default="label", choices=("label", "label_acute"))
    parser.add_argument("--overwrite", action="store_true", help="re-render existing PNGs")
    parser.add_argument(
        "--rendering", default="cohort_z", choices=sorted(RENDERINGS),
        help="cohort_z (default) or baseline_z (leaks the label -- inspection only)",
    )
    args = parser.parse_args(argv)

    if args.rendering == "baseline_z":
        print(
            "WARNING: baseline_z pins preflight samples near zero by construction.\n"
            "         Its magnitude control runs at AUROC 0.98-1.00 -- results on\n"
            "         this rendering are not reportable. See the module docstring.\n"
        )

    wanted = [args.dataset] if args.dataset else list(DATASETS)
    print(f"Rendering modalities into {OUT_ROOT.relative_to(REPO_ROOT)}/ ({args.rendering})")
    for dataset in wanted:
        try:
            render(dataset, args.label_col, args.overwrite, args.rendering)
        except FileNotFoundError as exc:
            print(f"  {dataset}: skipped ({exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
