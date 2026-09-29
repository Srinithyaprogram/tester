# Data sources

The proposal names four sources. They are **not** four independent corpora —
OSDR is the umbrella that now serves GeneLab and ALSDA, and the Inspiration4
data lives under the same accession scheme.

```
NASA OSDR  (osdr.nasa.gov)  ── the repository / API layer
   ├── GeneLab        space-biology omics
   ├── ALSDA          human & animal physiology / behaviour
   └── Inspiration4   OSD-569..575, OSD-630, OSD-656, OSD-687  (SOMA)
```

---

## Read this before designing the experiments

**The modality gap — RESOLVED, and it is bad news.** I audited the OSDR API on
30 July 2026. There is **no human movement assay in OSDR at all** — not gait, not
posturography, not actigraphy, not accelerometry. The complete list of human assay
types is molecular (transcription, methylation, protein, metabolite profiling)
plus a blood count and a metabolic panel.

Movement data *does* exist in OSDR, but only for **mouse and Drosophila**
(DigiGait, balance, open field). Human sensorimotor data lives in NASA's **LSDA**
instead, which is managed-access with a formal data-request process and no API.

Full evidence and the three ways forward: **`docs/OSDR_FINDINGS.md`**. Read it
before writing the abstract — it changes what this paper can be.

**The cohort-size problem.** Inspiration4 flew **four** people. Even pooling
across accessions, human spaceflight `n` is in the tens. This has hard
consequences:

- A subject-wise split over n=4 leaves ~1 crew member in test. Report
  leave-one-subject-out cross-validation and per-subject results, never a single
  pooled accuracy.
- Deep models will overfit on a cohort this size. That is itself a finding.
- Confidence intervals matter more than point estimates. `run.py` writes a
  `per_subject` column into `results/results.csv` for exactly this reason.

**Access.** Aggregate and derived data are open. Individual-level human records
(raw sequencing, identifiable clinical data) require an application with ethics
oversight board approval. Consequences for this repo:

- Never commit crew-level records. `.gitignore` already excludes `data/raw/`.
- If you attach controlled data as a Kaggle dataset, make it **private**.
- Cite the accession, not the file you downloaded.

---

## Getting the data

### 1. Discover what exists

```bash
python scripts/fetch_osdr.py --list-human          # all Homo sapiens accessions
python scripts/fetch_osdr.py --accession OSD-570   # metadata + file manifest
python scripts/fetch_osdr.py --inspiration4        # cache all I4 manifests
```

**Inspiration4 already has a real, rebuildable panel** — see
`scripts/fetch_inspiration4_files.py` (downloads the actual CBC / metabolic
panel / multiplex files for OSD-569, 570, 571, 575, 656) and
`scripts/build_inspiration4_panel.py` (merges them into
`data/raw/inspiration4/panel.csv`). It has real measures and no clinical
label — see `data/raw/EXAMPLE_DATA_README.md` for exactly what that means.

The API is documented at <https://visualization.osdr.nasa.gov/biodata/api/>.
Base URL: `https://visualization.osdr.nasa.gov/biodata/api/v2`.

Useful routes:

| Route | Returns |
|---|---|
| `/v2/datasets/` | every accession |
| `/v2/dataset/{acc}/` | study metadata |
| `/v2/dataset/{acc}/files/` | file manifest |
| `/v2/dataset/{acc}/assays/` | which measurement types exist |
| `/v2/query/samples/?<field>=<value>` | filtered sample table |

Filters use canonical metadata paths, e.g.
`?study.characteristics.organism=Homo%20sapiens`,
`?study.factor%20value.spaceflight=in-flight|pre-flight`. Add `&format=csv`
or `&format=json.records`.

### 2. Export a longitudinal panel

Every corpus reaches the models through one tidy CSV per dataset key:

```
data/raw/<dataset_key>/panel.csv
```

| Column | Required | Meaning |
|---|---|---|
| `subject_id` | yes | crew member identifier |
| `timepoint` | yes | `L-92`, `L-3`, `FD1`, `R+1`, `R+45`, ... |
| `phase` | no | `preflight` / `inflight` / `postflight` (inferred from `timepoint`) |
| `label` | **strongly recommended** | 1 = deteriorated |
| `split` | no | `train` / `val` / `test` (otherwise derived subject-wise) |
| *anything numeric* | — | treated as a measure |

See the format template:

```bash
python scripts/make_example_panel.py --dataset inspiration4 --n-subjects 4
```

Each sample becomes two channels: the cohort-standardised measurement vector,
and its deviation from **that crew member's own preflight baseline**. Own-baseline
referencing is what makes an n=4 cohort analysable at all — it removes
between-subject variation that would otherwise swamp the effect.

### 3. Supply a real label

If `panel.csv` has no `label` column, the loader derives one by thresholding the
baseline deviation — and then **warns loudly, marks the split
non-reportable, and prints a validity warning into the report**. This is
deliberate. That derived label is a function of the same features the model
receives, so a model scoring 1.000 on it has learned to recompute a threshold,
not to detect deterioration.

Valid label sources, in order of preference:

1. A clinical endpoint recorded in the study (diagnosis, flight-surgeon
   assessment, functional-test failure).
2. A pre-registered threshold on measures that are **excluded from the feature
   set** — e.g. label from bone density, predict from movement only.
3. Phase as a proxy (`inflight`/`early postflight` = exposed). Weaker, but
   honest and non-circular.

---

## Citing

| Source | Cite |
|---|---|
| OSDR | *NASA Open Science Data Repository: open science for life in space*, Nucleic Acids Research 53(D1):D1697 |
| Inspiration4 / SOMA | Overbey et al., *The Space Omics and Medical Atlas (SOMA) and international astronaut biobank*, Nature 632:1145–1154 (2024) |
| I4 data access | *Inspiration4 data access through the NASA Open Science Data Repository*, npj Microgravity (2024), doi:10.1038/s41526-024-00393-5 |
| Individual studies | the OSD accession, e.g. `OSD-570`, <https://osdr.nasa.gov/bio/repo/data/studies/OSD-570> |

BibTeX keys: `scott_osdr_2025`, `overbey_soma_2024`, `genelab_2019`, `alsda`.
