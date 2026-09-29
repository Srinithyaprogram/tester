#!/usr/bin/env python3
"""
Turn the two downloaded OSDR transcriptomics cohorts into tidy panel.csv files.

    genelab_astronaut   OSD-174  10 ISS astronauts x 6 timepoints (hair follicle)
    osdr_bedrest        OSD-370  10 bed-rest subjects (vastus lateralis)

    python scripts/fetch_osdr_transcriptomics.py   # first
    python scripts/build_osdr_panels.py

Labels come straight from NASA's own deposited factor values -- OSD-174's
Factor Value[Spaceflight] (pre-flight / in-flight / post-flight) and OSD-370's
Factor Value[Bed Rest] (Before / During / After Bed Rest). The baseline level
maps to 0, every other level to 1. Nothing is thresholded off the measurements
themselves, so unlike the baseline-deviation fallback in datasets.py this label
is NOT circular: it is mission metadata, independent of the feature matrix.

A stricter `label_acute` column is also written, which drops the ambiguous
recovery timepoints (post-flight, after-bed-rest-and-retraining) to -1 so they
can be excluded rather than silently counted as deteriorated.

Feature reduction: microarray platforms carry ~45k probes against 60 samples.
The panel keeps the --top-features most variable non-control probes, named by
gene symbol where the platform annotation resolves one. The full matrix is not
kept -- 45k columns is unusable for both the heatmap renderer (which caps at 64)
and any probe fit on n=60.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "data" / "raw" / "osdr_genelab_src"
RAW_ROOT = REPO_ROOT / "data" / "raw"

DEFAULT_TOP_FEATURES = 1024

#: Agilent control probes -- constant by construction, never a measurement.
CONTROL_PREFIXES = ("GE_BrightCorner", "DarkCorner", "(+)E1A", "ETG", "3xSLv1")

#: NASA factor level -> (label, is_recovery, canonical phase).
#:
#: The phase name is the repo's own vocabulary (datasets.py::_infer_phase and
#: validate_panel.py both key off preflight/inflight/postflight), not NASA's
#: wording -- the original string is preserved in the `phase_source` column.
#: Bed rest maps onto the same three-phase shape by analogy: the pre-exposure
#: measurement is the subject's own baseline, "during" is the exposed state,
#: and the post-retraining follow-up is the recovery arm.
OSD174_PHASES = {
    "pre-flight": (0, False, "preflight"),
    "in-flight": (1, False, "inflight"),
    "post-flight": (1, True, "postflight"),
}
OSD370_PHASES = {
    "Before Bed Rest": (0, False, "preflight"),
    "During Bed Rest": (1, False, "inflight"),
    "After Bed Rest and Retraining": (1, True, "postflight"),
}


def _read_isa(zip_path: Path, prefix: str) -> pd.DataFrame:
    """Read the one ISA-Tab table in `zip_path` whose name starts with `prefix`."""
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.startswith(prefix)]
        if not names:
            raise FileNotFoundError(f"no {prefix}* table inside {zip_path}")
        payload = archive.read(names[0])
    return pd.read_csv(io.BytesIO(payload), sep="\t", dtype=str)


def _probe_symbols() -> dict[str, str]:
    """Agilent probe ID -> gene symbol, borrowed from OSD-370's annotation.

    OSD-174 deposits per-sample tables keyed by bare Reporter Identifier with no
    platform annotation. Both studies run Agilent WholeGenome 4x44k (v2 and v1),
    whose probe IDs share a namespace, and OSD-370's GeneLab-processed matrix
    ships a ProbeName -> SYMBOL column. Unmapped probes keep their probe ID.
    """
    path = SRC_ROOT / "OSD-370" / "GLDS-370_array_normalized_expression_GLmicroarray.csv"
    if not path.exists():
        return {}
    annotation = pd.read_csv(path, usecols=["ProbeName", "SYMBOL"], low_memory=False)
    annotation = annotation.dropna(subset=["SYMBOL"])
    annotation = annotation[annotation["SYMBOL"].astype(str).str.strip().ne("")]
    return dict(zip(annotation["ProbeName"].astype(str), annotation["SYMBOL"].astype(str)))


def _select_features(
    matrix: pd.DataFrame, top_features: int
) -> tuple[pd.DataFrame, list[str]]:
    """Keep the `top_features` highest-variance columns of a samples x probes frame."""
    numeric = matrix.apply(pd.to_numeric, errors="coerce")
    # A probe is unusable if it is missing anywhere or never varies.
    usable = numeric.columns[numeric.notna().all(axis=0) & (numeric.std(axis=0) > 1e-8)]
    numeric = numeric[usable]
    if numeric.shape[1] > top_features:
        keep = numeric.std(axis=0).nlargest(top_features).index
        # Preserve platform order rather than variance order, for readability.
        keep = [c for c in numeric.columns if c in set(keep)]
        numeric = numeric[keep]
    return numeric, list(numeric.columns)


def _finalise(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    dataset_key: str,
    source: str,
) -> Path:
    """Write data/raw/<dataset_key>/panel.csv from aligned metadata + features."""
    panel = pd.concat([meta.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    out_dir = RAW_ROOT / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "panel.csv"
    panel.to_csv(out_path, index=False)

    n_pos = int((panel["label"] == 1).sum())
    n_acute = int((panel["label_acute"] >= 0).sum())
    print(
        f"  {dataset_key}: {panel.shape[0]} samples x {features.shape[1]} features"
        f" | {panel['subject_id'].nunique()} subjects"
        f" | label 1/0 = {n_pos}/{len(panel) - n_pos}"
        f" | label_acute keeps {n_acute}"
    )
    print(f"  -> {out_path.relative_to(REPO_ROOT)}  ({source})")
    return out_path


def build_osd174(top_features: int) -> Path:
    """OSD-174 -- 10 ISS astronauts, hair-follicle microarray, 6 timepoints each."""
    src = SRC_ROOT / "OSD-174"
    study = _read_isa(src / "OSD-174-ISA.zip", "s_")
    assay = _read_isa(src / "OSD-174-ISA.zip", "a_")

    # Sample Name is the join key; Hybridization Assay Name is the GSM the
    # per-sample processed table is filed under.
    key = assay[["Sample Name", "Hybridization Assay Name"]].rename(
        columns={"Hybridization Assay Name": "gsm"}
    )
    study = study.merge(key, on="Sample Name", how="left")
    if study["gsm"].isna().any():
        raise ValueError("OSD-174: some samples have no GSM in the assay table")

    with zipfile.ZipFile(src / "processed.zip") as archive:
        tables: dict[str, pd.Series] = {}
        for name in archive.namelist():
            gsm = name.split("_")[0]
            frame = pd.read_csv(io.BytesIO(archive.read(name)), sep="\t")
            tables[gsm] = frame.set_index("Reporter Identifier")["VALUE"]
    matrix = pd.DataFrame(tables).T  # samples x probes
    matrix = matrix.drop(
        columns=[c for c in matrix.columns if str(c).startswith(CONTROL_PREFIXES)],
        errors="ignore",
    )
    # Name probes by gene symbol so the text modality and any biomedical text
    # encoder see vocabulary they were pretrained on, not bare Agilent IDs.
    symbols = _probe_symbols()
    matrix.columns = [
        f"{symbols[p]}|{p}" if p in symbols else p for p in matrix.columns.astype(str)
    ]
    matrix = matrix.loc[:, ~matrix.columns.duplicated()]

    missing = set(study["gsm"]) - set(matrix.index)
    if missing:
        raise ValueError(f"OSD-174: {len(missing)} samples absent from processed.zip")
    matrix = matrix.loc[study["gsm"].to_numpy()]

    phases = study["Factor Value[Spaceflight]"].map(OSD174_PHASES)
    meta = pd.DataFrame(
        {
            "subject_id": study["Source Name"].to_numpy(),
            "timepoint": study["Factor Value[Time]"].to_numpy(),
            "phase": [p[2] for p in phases],
            "phase_source": study["Factor Value[Spaceflight]"].to_numpy(),
            "sex": study["Factor Value[Sex]"].to_numpy(),
            "label": [p[0] for p in phases],
            "label_acute": [-1 if p[1] else p[0] for p in phases],
        }
    )
    features, _ = _select_features(matrix, top_features)
    return _finalise(features, meta, "genelab_astronaut", "OSD-174, hair follicle microarray")


def build_osd370(top_features: int) -> Path:
    """OSD-370 -- 10 bed-rest subjects, vastus lateralis microarray."""
    src = SRC_ROOT / "OSD-370"
    study = _read_isa(src / "OSD-370-ISA.zip", "s_")

    expression = pd.read_csv(
        src / "GLDS-370_array_normalized_expression_GLmicroarray.csv", low_memory=False
    )
    sample_cols = [c for c in expression.columns if c.startswith("GSM")]
    # Name probes by gene symbol where the platform annotation resolves one,
    # de-duplicated so several probes per gene stay distinguishable.
    symbol = expression["SYMBOL"].fillna("").astype(str)
    probe = expression["ProbeName"].astype(str)
    names = np.where(symbol.ne("") & symbol.ne("nan"), symbol + "|" + probe, probe)
    matrix = pd.DataFrame(
        expression[sample_cols].to_numpy().T, index=sample_cols, columns=names
    )
    matrix = matrix.loc[:, ~matrix.columns.duplicated()]

    study = study[study["Sample Name"].isin(matrix.index)].reset_index(drop=True)
    matrix = matrix.loc[study["Sample Name"].to_numpy()]

    phases = study["Factor Value[Bed Rest]"].map(OSD370_PHASES)
    meta = pd.DataFrame(
        {
            "subject_id": "BR-" + study["Characteristics[Study Subject]"].str.zfill(2),
            "timepoint": study["Factor Value[Time]"].to_numpy(),
            "phase": [p[2] for p in phases],
            "phase_source": study["Factor Value[Bed Rest]"].to_numpy(),
            "treatment": study["Factor Value[Treatment]"].to_numpy(),
            "sex": study["Characteristics[Sex]"].to_numpy(),
            "label": [p[0] for p in phases],
            "label_acute": [-1 if p[1] else p[0] for p in phases],
        }
    )
    features, _ = _select_features(matrix, top_features)
    return _finalise(features, meta, "osdr_bedrest", "OSD-370, vastus lateralis microarray")


BUILDERS = {"genelab_astronaut": build_osd174, "osdr_bedrest": build_osd370}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(BUILDERS), help="build only one panel")
    parser.add_argument("--top-features", type=int, default=DEFAULT_TOP_FEATURES)
    args = parser.parse_args(argv)

    wanted = [args.dataset] if args.dataset else sorted(BUILDERS)
    print("Building panels from data/raw/osdr_genelab_src/")
    for key in wanted:
        try:
            BUILDERS[key](args.top_features)
        except FileNotFoundError as exc:
            print(f"  {key}: source missing ({exc})")
            print("  run: python scripts/fetch_osdr_transcriptomics.py")
            return 1
    print("Validate with: python scripts/validate_panel.py --dataset <key>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
