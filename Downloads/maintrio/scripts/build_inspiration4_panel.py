#!/usr/bin/env python3
"""
Build a real Inspiration4 panel.csv from the OSDR clinical/immunoassay files
downloaded into data/raw/inspiration4_osdr/ (see docs/DATASETS.md).

Merges five per-accession CSVs -- complete blood count (OSD-569), comprehensive
metabolic panel, cardiovascular and immune multiplex panels (OSD-575) -- on
(subject_id, timepoint). Each source's reference-range columns (which are
constant, not measurements) are dropped; only the actual value/concentration/
percent columns are kept, prefixed by source so overlapping analyte names
across panels never collide.

    python scripts/build_inspiration4_panel.py

Real NASA data in, but note: there is still no clinical 'label' column in any
of these files, so datasets.py will derive one from baseline deviation and
flag it circular -- see docs/DATASETS.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OSDR_ROOT = REPO_ROOT / "data" / "raw" / "inspiration4_osdr"

SOURCES = {
    "cbc": OSDR_ROOT / "OSD-569" / "LSDS-7_Complete_Blood_Count_CBC_TRANSFORMED.csv",
    "cmp": OSDR_ROOT / "OSD-575" / "LSDS-8_Comprehensive_Metabolic_Panel_CMP_TRANSFORMED.csv",
    "cardio": OSDR_ROOT / "OSD-575" / "LSDS-8_Multiplex_serum_cardiovascular_EvePanel_TRANSFORMED.csv",
    "immune_eve": OSDR_ROOT / "OSD-575" / "LSDS-8_Multiplex_serum_immune_EvePanel_TRANSFORMED.csv",
    "immune_alamar": OSDR_ROOT / "OSD-575" / "LSDS-8_Multiplex_serum.immune.AlamarPanel_TRANSFORMED.csv",
}
#: Two CMP reference-range-max columns that don't follow the 'range' naming
#: convention the rest of the source data uses -- exclude by name explicitly.
NON_MEASURE_COLUMNS = {"alt_max_units_per_liter", "ast_max_units_per_liter"}
MEASURE_MARKERS = ("_value", "_concentration_", "_percent")


def _load_source(tag: str, path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_col = "Sample Name" if "Sample Name" in frame.columns else "Sample ID"
    tokens = frame[id_col].str.split("_")
    frame["subject_id"] = tokens.str[0]
    frame["timepoint"] = tokens.str[2]  # e.g. C001_serum_L-3[_suffix] -> "L-3"

    measure_cols = [
        c for c in frame.columns
        if c not in (id_col, "subject_id", "timepoint")
        and c not in NON_MEASURE_COLUMNS
        and "range_min" not in c and "range_max" not in c
        and any(marker in c for marker in MEASURE_MARKERS)
    ]
    frame = frame[["subject_id", "timepoint"] + measure_cols]
    frame = frame.rename(columns={c: f"{tag}__{c}" for c in measure_cols})
    return frame.groupby(["subject_id", "timepoint"], as_index=False).first()


#: Timepoints OSDR files under Factor Value[Spaceflight] = "Post-flight" but
#: which are months downstream of return -- physiologically a recovery arm, not
#: an exposed one. Kept out of `label_acute` rather than silently counted as
#: deteriorated. R+45 stays exposed: most I4 crew markers had not renormalised
#: by then (Overbey et al., Nature 2024).
RECOVERY_TIMEPOINTS = {"R+82", "R+194"}


def _phase_labels(timepoint: pd.Series) -> pd.DataFrame:
    """Flight-phase label from the mission timeline, not from the measurements.

    L-<n> is a preflight baseline draw, R+<n> a postflight one. That is mission
    metadata and completely independent of the assay values, so unlike the
    baseline-deviation fallback in datasets.py this label is not circular.
    It is still an exposure proxy, not a clinical deterioration diagnosis --
    say so wherever it is reported.
    """
    tp = timepoint.astype(str).str.strip().str.upper()
    is_post = tp.str.startswith("R+") | tp.str.startswith("R-")
    return pd.DataFrame(
        {
            "phase": np.where(is_post, "postflight", "preflight"),
            "phase_source": np.where(is_post, "post-flight", "pre-flight"),
            "label": is_post.astype(int),
            "label_acute": np.where(
                tp.isin(RECOVERY_TIMEPOINTS), -1, is_post.astype(int)
            ),
        }
    )


def main() -> int:
    missing = [path for path in SOURCES.values() if not path.exists()]
    if missing:
        print("Missing source file(s) -- run scripts/fetch_osdr.py --inspiration4 "
              "then scripts/fetch_inspiration4_files.py first:")
        for path in missing:
            print(f"  {path}")
        return 1

    frames = [_load_source(tag, path) for tag, path in SOURCES.items()]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["subject_id", "timepoint"], how="outer")

    labels = _phase_labels(merged["timepoint"])
    measures = [c for c in merged.columns if c not in ("subject_id", "timepoint")]
    merged = pd.concat(
        [merged[["subject_id", "timepoint"]], labels, merged[measures]], axis=1
    )

    destination = REPO_ROOT / "data" / "raw" / "inspiration4" / "panel.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(destination, index=False)

    n_pos = int(labels["label"].sum())
    print(f"wrote {destination}")
    print(f"  {merged.shape[0]} rows | {merged['subject_id'].nunique()} crew members | "
          f"{merged['timepoint'].nunique()} timepoints | {len(measures)} measures")
    print(f"  label 1/0 = {n_pos}/{len(labels) - n_pos} | "
          f"label_acute keeps {int((labels['label_acute'] >= 0).sum())}")
    print("\nREAL DATA (OSDR OSD-569 / OSD-575). The label is the flight phase from")
    print("the mission timeline -- an exposure proxy, not a clinical diagnosis, but")
    print("independent of the assay values, so it is not circular.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
