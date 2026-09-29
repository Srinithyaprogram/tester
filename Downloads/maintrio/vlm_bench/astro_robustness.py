#!/usr/bin/env python3
"""
Do the benchmark's conclusions survive the stricter label?

Every table elsewhere uses `label`, which follows NASA's deposited factor value:
baseline level 0, every other level 1. That lumps genuinely ambiguous rows in
with the exposed class -- Inspiration4's R+82 and R+194 draws are six months
after return, OSD-174's post-flight arm is up to 90 days after landing, and
OSD-370's "after bed rest and retraining" arm is a recovery measurement.

`label_acute` drops those rows rather than counting them as deteriorated. This
script re-scores every model on that subset.

    python vlm_bench/astro_robustness.py

Writes vlm_bench/results/robustness_label_acute.csv.

**Method note.** The encodings are reused and subset, not regenerated. A true
label_acute rendering would z-score against the smaller cohort, so its heatmaps
would differ slightly from the ones actually encoded here. Re-encoding six
models x three corpora on the GPU box to chase that second-order difference was
not worth it; what this measures is whether the *conclusions* survive dropping
the ambiguous rows, and for that the existing embeddings are the right input --
the model saw an identical image in both conditions, so any change is
attributable to the label, not to the rendering.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from astro_eval import (  # noqa: E402
    DATASET_PRETTY, DATASETS, PRETTY, linear_probe_loso, load_encoding, metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
RESULTS = Path(__file__).resolve().parent / "results"

MODELS = ("clip", "blip2", "llava16", "qwen25vl", "biomedclip", "medflamingo",
          "biovilt", "chexagent")


def acute_mask(dataset: str, n_expected: int) -> np.ndarray | None:
    """Boolean mask over the encoding's samples: True where label_acute >= 0.

    astro_modalities renders in panel order after filtering `label >= 0` (which
    keeps every row), and encoding preserves that order, so panel row i is
    sample i. Verified by length and by the label vector matching.
    """
    path = RAW_ROOT / dataset / "panel.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path, low_memory=False)
    if "label_acute" not in frame.columns or len(frame) != n_expected:
        return None
    return (frame["label_acute"] >= 0).to_numpy()


def main() -> int:
    rows = []
    for dataset in DATASETS:
        reference = load_encoding(dataset, "clip")
        if reference is None:
            continue
        mask = acute_mask(dataset, len(reference["labels"]))
        if mask is None:
            print(f"  {dataset}: cannot align panel to encodings, skipped")
            continue

        frame = pd.read_csv(RAW_ROOT / dataset / "panel.csv", low_memory=False)
        if not np.array_equal(frame["label"].to_numpy(), reference["labels"]):
            print(f"  {dataset}: panel/encoding label mismatch, skipped")
            continue

        for model in MODELS:
            data = load_encoding(dataset, model)
            if data is None:
                continue
            y_full = data["labels"]
            for protocol in ("zero_shot", "linear_probe_loso"):
                for condition, keep in (("label", np.ones_like(mask)), ("label_acute", mask)):
                    y = y_full[keep]
                    subjects = data["subjects"][keep]
                    if len(np.unique(y)) < 2:
                        continue
                    if protocol == "zero_shot":
                        score = data["zeroshot_score"][keep]
                        pred = (score >= 0.5).astype(int)
                    else:
                        pred, score = linear_probe_loso(
                            data["image_embed"][keep], y, subjects
                        )
                    rows.append({
                        "model": PRETTY.get(model, model),
                        "dataset": DATASET_PRETTY.get(dataset, dataset),
                        "protocol": protocol,
                        "label_col": condition,
                        "n_samples": int(keep.sum()),
                        **metrics(y, score, pred),
                    })

    if not rows:
        print("nothing to score")
        return 1

    frame = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "robustness_label_acute.csv"
    frame.to_csv(out, index=False)

    # Side-by-side on the metric the write-up leads with.
    probe = frame[frame.protocol == "linear_probe_loso"]
    pivot = probe.pivot_table(
        index=["dataset", "model"], columns="label_col",
        values=["balanced_accuracy", "mcc"],
    )
    with pd.option_context("display.width", 200):
        print(pivot.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
