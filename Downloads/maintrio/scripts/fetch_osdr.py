#!/usr/bin/env python3
"""
Discover and cache NASA OSDR metadata (GeneLab / ALSDA / Inspiration4).

Run this once with network access, commit the cache, and the experiment scripts
then work in an offline Kaggle kernel.

    python scripts/fetch_osdr.py --list-human
    python scripts/fetch_osdr.py --accession OSD-570
    python scripts/fetch_osdr.py --inspiration4
    python scripts/fetch_osdr.py --search-assays gait balance sway locomotion actigraphy

The last form is the one to run first: it tells you whether the archive actually
holds movement/behavioural assays, which determines whether the movement framing
in the proposal survives contact with the data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import osdr_client as osdr  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list-human", action="store_true",
                        help="List every Homo sapiens accession in OSDR.")
    parser.add_argument("--accession", help="Fetch metadata + file manifest for one OSD-xxx.")
    parser.add_argument("--inspiration4", action="store_true",
                        help="Cache manifests for all known Inspiration4 accessions.")
    parser.add_argument("--search-assays", nargs="+", metavar="TERM",
                        help="Report which human accessions have assays matching these terms.")
    parser.add_argument("--data-root", default=None, help="Defaults to <repo>/data")
    args = parser.parse_args()

    data_root = Path(args.data_root) if args.data_root else REPO_ROOT / "data"
    cache = data_root / "external" / "osdr"
    cache.mkdir(parents=True, exist_ok=True)

    if args.list_human:
        studies = osdr.list_human_studies()
        path = cache / "human_studies.json"
        path.write_text(json.dumps(studies, indent=2), encoding="utf-8")
        print(f"{len(studies)} human accessions -> {path}")

    if args.accession:
        payload = {
            "url": osdr.study_url(args.accession),
            "metadata": osdr.study_metadata(args.accession),
            "assays": osdr.study_assays(args.accession),
            "files": osdr.study_files(args.accession),
        }
        path = cache / f"{args.accession}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"{args.accession} -> {path}")
        print(f"  landing page: {payload['url']}")

    if args.inspiration4:
        path = osdr.cache_manifest(osdr.INSPIRATION4_ACCESSIONS, cache)
        print(f"cached {len(osdr.INSPIRATION4_ACCESSIONS)} I4 manifests -> {path}")

    if args.search_assays:
        terms = [t.lower() for t in args.search_assays]
        print(f"searching human accessions for assays matching: {', '.join(terms)}\n")
        hits = 0
        for record in osdr.list_human_studies():
            accession = record.get("id.accession") or record.get("accession") or str(record)
            try:
                assays = osdr.study_assays(accession)
            except Exception as exc:  # noqa: BLE001 — restricted or transient
                print(f"  {accession}: unavailable ({type(exc).__name__})")
                continue
            blob = json.dumps(assays).lower()
            matched = [t for t in terms if t in blob]
            if matched:
                hits += 1
                print(f"  {accession}: {', '.join(matched)}  {osdr.study_url(accession)}")
        print(f"\n{hits} accession(s) matched. "
              f"If this is 0, see the 'modality gap' section of docs/DATASETS.md.")

    if not any([args.list_human, args.accession, args.inspiration4, args.search_assays]):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
