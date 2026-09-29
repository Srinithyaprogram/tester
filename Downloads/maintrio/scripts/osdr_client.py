"""
Thin client for the NASA Open Science Data Repository (OSDR) Biological Data API.

OSDR is the umbrella repository serving both GeneLab (space biology omics) and
the Ames Life Sciences Data Archive (ALSDA). Inspiration4 (I4) is published
under the same accession scheme (OSD-569..575, plus OSD-630/656/687).

API reference: https://visualization.osdr.nasa.gov/biodata/api/

Access note: aggregate/derived data are open; individual-level human data
(raw sequencing, identifiable clinical records) requires controlled-access
approval. Never commit crew-level records to the repo.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API_ROOT = "https://visualization.osdr.nasa.gov/biodata/api/v2"
STUDY_PAGE = "https://osdr.nasa.gov/bio/repo/data/studies/{accession}"

#: Inspiration4 accessions confirmed in the OSDR data-access paper
#: (npj Microgravity, 2024; doi:10.1038/s41526-024-00393-5).
INSPIRATION4_ACCESSIONS: tuple[str, ...] = (
    "OSD-569", "OSD-570", "OSD-571", "OSD-572", "OSD-573",
    "OSD-574", "OSD-575", "OSD-630", "OSD-656", "OSD-687",
)


def study_url(accession: str) -> str:
    return STUDY_PAGE.format(accession=accession)


def _get(path: str, timeout: float = 60.0, **params: Any) -> Any:
    query = urllib.parse.urlencode({"format": "json.records", **params}, safe="|")
    url = f"{API_ROOT}/{path.lstrip('/')}?{query}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def list_human_studies(**filters: Any) -> Any:
    """All OSDR datasets whose organism is *Homo sapiens*."""
    return _get("query/datasets/", **{"study.characteristics.organism": "Homo sapiens", **filters})


def study_metadata(accession: str) -> Any:
    return _get(f"dataset/{accession}/")


def study_files(accession: str) -> Any:
    return _get(f"dataset/{accession}/files/")


def study_assays(accession: str) -> Any:
    return _get(f"dataset/{accession}/assays/")


def cache_manifest(accessions: tuple[str, ...], destination: str | Path) -> Path:
    """Fetch and cache file manifests so later runs work offline."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {}
    for accession in accessions:
        try:
            manifest[accession] = {
                "url": study_url(accession),
                "assays": study_assays(accession),
                "files": study_files(accession),
            }
        except Exception as exc:  # noqa: BLE001 -- offline / restricted access
            manifest[accession] = {"url": study_url(accession), "error": str(exc)}
    path = destination / "osdr_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
