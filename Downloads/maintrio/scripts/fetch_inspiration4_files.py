#!/usr/bin/env python3
"""
Download the actual OSDR data files (not just manifests) for the small,
clinically-usable Inspiration4 accessions.

    python scripts/fetch_osdr.py --inspiration4   # cache the manifest first
    python scripts/fetch_inspiration4_files.py    # then pull the files

Scope, deliberately: only OSD-569, 570, 571, 575, 656 -- the five accessions
whose file lists are small (CBC, metabolic panel, proteomics/metabolomics,
immune/cardiovascular multiplex panels). OSD-572/573/574/630/687 are excluded
-- their "tabular" files are per-sample metagenomic-sequencing pipeline
output (thousands of files), not clinical measures, and would need a separate
bioinformatics workflow to turn into per-subject features. Also skipped:
plain '*_md5sum_*.txt' checksum listings (not data) and raw sequencing files
(fastq/bam/etc. -- not usable by this repo's tidy-panel pipeline, and easily
gigabytes each). See docs/OSDR_FINDINGS.md and docs/DATASET_SUMMARY.md.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "external" / "osdr" / "osdr_manifest.json"
DESTINATION_ROOT = REPO_ROOT / "data" / "raw" / "inspiration4_osdr"

ACCESSIONS = ("OSD-569", "OSD-570", "OSD-571", "OSD-575", "OSD-656")
TABULAR_EXTENSIONS = (".csv", ".tsv", ".xlsx", ".txt", ".json")


def _is_safe_filename(name: str) -> bool:
    """Reject anything that isn't a single, plain filename.

    The manifest is API-sourced JSON cached to disk -- treat it as untrusted
    input, not as a trusted local file, since it could be edited, shared, or
    regenerated from a tampered response independently of this script.
    """
    return PurePosixPath(name).name == name and name not in ("", ".", "..")


def select_targets(manifest: dict) -> list[tuple[str, str, str]]:
    """(accession, filename, url) for every non-checksum tabular file."""
    targets = []
    for accession in ACCESSIONS:
        files = manifest[accession]["files"][accession]["files"]
        for name, info in files.items():
            if not name.lower().endswith(TABULAR_EXTENSIONS):
                continue
            if "md5sum" in name.lower():
                continue
            if not _is_safe_filename(name):
                print(f"SKIP  {accession}/{name}  (unsafe filename, refusing to write)")
                continue
            url = info["URL"]
            if not url.startswith("https://"):
                print(f"SKIP  {accession}/{name}  (non-https URL, refusing to fetch: {url})")
                continue
            targets.append((accession, name, url))
    return targets


def main() -> int:
    if not MANIFEST_PATH.exists():
        print(f"No manifest at {MANIFEST_PATH} -- run "
              f"'python scripts/fetch_osdr.py --inspiration4' first.")
        return 1

    manifest = json.loads(MANIFEST_PATH.read_text())
    targets = select_targets(manifest)
    print(f"{len(targets)} file(s) selected across {len(ACCESSIONS)} accessions.\n")

    ok = 0
    for accession, name, url in targets:
        destination = DESTINATION_ROOT / accession / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "ahm-research/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                data = response.read()
            destination.write_bytes(data)
            print(f"OK    {accession}/{name}  ({len(data):,} bytes)")
            ok += 1
        except Exception as exc:  # noqa: BLE001 -- report and keep going
            print(f"FAIL  {accession}/{name}  ({type(exc).__name__}: {exc})")

    print(f"\n{ok}/{len(targets)} downloaded to {DESTINATION_ROOT}")
    print("Next: python scripts/build_inspiration4_panel.py")
    return 0 if ok == len(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
