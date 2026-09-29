#!/usr/bin/env python3
"""
Download the two OSDR transcriptomics cohorts that have per-sample processed
data AND a subject identifier, so they can become tidy panels.

    OSD-174  10 ISS astronauts x 6 timepoints, hair-follicle microarray
             (pre-flight / in-flight / post-flight)
    OSD-370  10 bed-rest subjects, vastus lateralis microarray
             (before / during / after bed rest)

Both are Agilent WholeGenome 4x44k, so their probe IDs share a namespace --
that is what makes a gene-symbol-level comparison between them possible.

    python scripts/fetch_osdr_transcriptomics.py

Downloads ~80MB into data/raw/osdr_genelab_src/. Already-present files are
skipped, so this is safe to re-run. Build the panels with
scripts/build_osdr_panels.py.

Why not the other candidates: OSD-530 (6 JAXA astronauts, cell-free RNA) only
deposits group-level means and SEM -- no per-sample matrix short of running an
RNA-seq pipeline from fastq. OSD-483 deposits fastq only.
"""

from __future__ import annotations

import argparse
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEST_ROOT = REPO_ROOT / "data" / "raw" / "osdr_genelab_src"

DOWNLOAD = "https://osdr.nasa.gov/geode-py/ws/studies/{accession}/download?source=datamanager&file={name}"

#: (accession, remote file name, local name)
FILES: tuple[tuple[str, str, str], ...] = (
    (
        "OSD-174",
        "OSD-174_metadata_E-GEOD-74708-ISA.zip",
        "OSD-174-ISA.zip",
    ),
    (
        "OSD-174",
        "GLDS-174_microarray_GLDS-174_microarray_E-GEOD-74708.processed.1.zip",
        "processed.zip",
    ),
    (
        "OSD-370",
        "OSD-370_metadata_OSD-370-ISA.zip",
        "OSD-370-ISA.zip",
    ),
    (
        "OSD-370",
        "GLDS-370_array_normalized_expression_GLmicroarray.csv",
        "GLDS-370_array_normalized_expression_GLmicroarray.csv",
    ),
)


def fetch(accession: str, remote: str, local: str, timeout: float = 900.0) -> Path:
    out = DEST_ROOT / accession / local
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        print(f"  cached  {accession}/{local} ({out.stat().st_size / 1e6:.1f} MB)")
        return out
    url = DOWNLOAD.format(accession=accession, name=urllib.parse.quote(remote))
    print(f"  fetch   {accession}/{local} ...", end="", flush=True)
    request = urllib.request.Request(url, headers={"Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        payload = response.read()
    out.write_bytes(payload)
    print(f" {len(payload) / 1e6:.1f} MB")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accession",
        choices=sorted({a for a, _, _ in FILES}),
        help="fetch only one accession (default: both)",
    )
    args = parser.parse_args(argv)

    wanted = [f for f in FILES if args.accession in (None, f[0])]
    print(f"Downloading {len(wanted)} file(s) into {DEST_ROOT}")
    for accession, remote, local in wanted:
        try:
            fetch(accession, remote, local)
        except Exception as exc:  # noqa: BLE001 -- offline / OSDR outage
            print(f"\n  FAILED  {accession}/{local}: {exc}")
            return 1
    print("Done. Next: python scripts/build_osdr_panels.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
