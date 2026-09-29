# Datasets — Astronaut Health Monitoring (ICDM 2026 Workshop)

Data sources wired into the experiment infrastructure, with current status.

**Status key:** ✅ working · ⬜ registered, not yet downloaded · ⚠️ placeholder data present

---

## 1. NASA Open Science Data Repository (OSDR) ⬜

- **Registry key:** `osdr_human_spaceflight`
- **Link:** https://osdr.nasa.gov/bio/repo/
- **API:** https://visualization.osdr.nasa.gov/biodata/api/v2
- **Scope:** ~121 *Homo sapiens* accessions (verified against the API, July 2026), accessions `OSD-2` … `OSD-1005`
- **Modality:** longitudinal biomedical panels — omics plus clinical assays
- **Access:** aggregate/derived data open; individual-level human records require controlled-access approval from an ethics oversight board
- **Cite:** *NASA Open Science Data Repository: open science for life in space*, Nucleic Acids Research 53(D1):D1697

OSDR is the **umbrella repository**, not a sibling of the three below. GeneLab and ALSDA are collections served through it, and the Inspiration4 data uses the same accession scheme. Worth stating plainly in the paper — a reviewer reading "four datasets" would be reading an overclaim.

## 2. NASA GeneLab ⬜

- **Registry key:** `genelab_omics`
- **Link:** https://genelab.nasa.gov/
- **Modality:** multi-omics — transcriptomic, proteomic, metabolomic
- **Access:** open; human sequencing requires controlled access

Molecular, not movement. Supports the *physiological* deterioration arm of the study, not the gait/posture arm. Expect n_features ≫ n_subjects — reduce dimensionality before modelling and report per-subject variance.

## 3. Ames Life Sciences Data Archive (ALSDA) ⚠️

- **Registry key:** `alsda_physiology`
- **Link:** https://osdr.nasa.gov/bio/repo/ (ALSDA-sourced accessions)
- **Modality:** physiological and behavioural measures from spaceflight and ground analogues (bed-rest, isolation)
- **Access:** open; identifiable human records controlled-access

**Audited — and the movement data is not here.** ALSDA-sourced human accessions in OSDR expose no gait, balance or actigraphy assay. NASA's separate LSDA archive (nlsp.nasa.gov) does hold "Behavior and performance" and "Neuroscience" data for astronauts, but it is managed-access via a formal data request, with no public API. See `docs/OSDR_FINDINGS.md`.

*Currently holds a synthetic placeholder panel generated to test the pipeline. Not real data.*

## 4. Inspiration4 / SOMA ✅ (inputs) / ⚠️ (labels)

- **Registry key:** `inspiration4`
- **Link:** https://osdr.nasa.gov/bio/repo/data/studies/OSD-570
- **Accessions:** `OSD-569`, `570`, `571`, `572`, `573`, `574`, `575`, `630`, `656`, `687`
- **Cohort:** **4 crew members**, 3-day free-flying orbital mission (September 2021)
- **Timepoints:** preflight (L-92, L-44, L-3), inflight, postflight (R+1, R+45, R+82, R+194)
- **Modality:** multimodal biomedical — omics, clinical panels, biometrics
- **Access:** open aggregate; individual-level requires ethics-board approval
- **Cite:** Overbey et al., *The Space Omics and Medical Atlas (SOMA) and international astronaut biobank*, Nature 632:1145–1154 (2024). Access paper: npj Microgravity (2024), doi:10.1038/s41526-024-00393-5

*Real data as of this pull: complete blood count and comprehensive metabolic
panel (OSD-569, OSD-575), plus immune/cardiovascular multiplex panels
(OSD-575, 656) — 4 crew x 7 timepoints, 605 measures, merged by
`scripts/build_inspiration4_panel.py`. Proteomics/metabolomics (OSD-571) and
single-cell RNA/ATAC-seq (OSD-570) were downloaded but not yet merged in —
thousands of analytes/genes each, need dimensionality reduction first. Raw
sequencing and the metagenomic accessions (572/573/574/630/687) were not
pulled — not usable by this pipeline without a separate bioinformatics
workflow. No clinical `label` column exists in any of it — the derived,
circular label still applies. See `data/raw/EXAMPLE_DATA_README.md`.*

## 5. NASA OSDR — rodent behavioural assays ⬜ *(added after the audit)*

- **Registry key:** `osdr_rodent_behaviour`
- **Link:** https://osdr.nasa.gov/bio/repo/
- **Modality:** DigiGait gait kinematics, balance, locomotory counts, open field
- **Species:** mouse and *Drosophila* — **not human**
- **Access:** fully open, no application

The only openly-downloadable movement data in OSDR. Relevant if you take option B.

## 6. Synthetic Crew Gait ✅

- **Registry key:** `synthetic_crew_gait`
- **Generated in code** — no download, no files
- 24 simulated crew, 6-channel IMU (tri-axial accelerometer + gyroscope), 10 s windows at 50 Hz
- Deterioration modelled as reduced cadence, amplified medio-lateral sway, elevated stride-time variability
- Tuned to land at ~0.80–0.88 balanced accuracy so pipeline breakage is visible

Smoke test only. Never report it.

---

## Common format

Every repository corpus reaches the models through one tidy CSV at `data/raw/<key>/panel.csv`:

| Column | Required | Meaning |
|---|---|---|
| `subject_id` | yes | crew member identifier |
| `timepoint` | yes | `L-92`, `L-3`, `FD1`, `R+1`, `R+45`, … |
| `phase` | no | `preflight` / `inflight` / `postflight` (inferred from `timepoint`) |
| `label` | **strongly recommended** | 1 = deteriorated |
| `split` | no | `train` / `val` / `test` (otherwise derived subject-wise) |
| *any numeric column* | — | treated as a measure |

Each sample becomes two channels: the cohort-standardised measurement vector, and its deviation from **that crew member's own preflight baseline**. Own-baseline referencing is what makes an n=4 cohort analysable — it removes between-subject variation that would otherwise swamp the effect.

---

## Three things that constrain the study design

**1. The movement modality is not in this data — audited and confirmed.**

OSDR's human accessions contain no gait, posture, balance, actigraphy or
accelerometry assay of any kind. Every human assay type is molecular, plus a blood
count and a metabolic panel. Verified against the OSDR API on 30 July 2026.

Movement assays in OSDR are **rodent and Drosophila only** (DigiGait, balance,
open field) — open access, registered here as `osdr_rodent_behaviour`. Human
sensorimotor data is held in NASA's **LSDA** (nlsp.nasa.gov), a managed-access
archive with a formal data-request process and no public API.

Three honest options, detailed in `docs/OSDR_FINDINGS.md`:

- **A.** Reframe to multimodal *physiological* deterioration on human OSDR / I4 data — start today, but drop "behavioural/movement" from the title.
- **B.** Rodent gait and balance — real kinematics, open access, better n. Start today, but argue translation rather than assert it.
- **C.** Request human movement data from LSDA — matches the proposal, but a request cycle may not fit the workshop deadline. Worth submitting in parallel either way.

A hybrid of B (movement result) and A (human transfer arm) is a complete story
with data downloadable today.

**2. Cohort sizes are tiny.** Inspiration4 flew four people; human spaceflight n is in the tens even pooled. Consequences: use leave-one-subject-out rather than a single held-out split, report per-subject results rather than a pooled accuracy, and expect deep models to overfit (which is itself a reportable finding).

**3. Labels must come from outside the feature set.** If a panel ships no `label` column, the pipeline derives one by thresholding baseline deviation — and then flags the split as non-reportable, because that label is a function of the same features the model receives. A model scoring 1.000 on it has learned to recompute a threshold, not to detect deterioration.

Valid label sources, best first:

1. A clinical endpoint recorded in the study (diagnosis, flight-surgeon assessment, functional-test failure)
2. A pre-registered threshold on measures **excluded** from the feature set — e.g. label from bone density, predict from movement
3. Phase as a proxy (inflight / early postflight = exposed). Weaker, but honest and non-circular

---

## Current state

| Source | Registered | Data present | Reportable |
|---|---|---|---|
| OSDR | ✅ | ✗ | ✗ |
| GeneLab | ✅ | ✗ | ✗ |
| ALSDA | ✅ | placeholder only | ✗ |
| Inspiration4 | ✅ | **real** (CBC, CMP, multiplex panels) | ✗ — no clinical label yet |
| OSDR rodent behaviour | ✅ | ✗ | available, needs download |
| Synthetic | ✅ | ✅ | ✗ (by design) |

All specs, source links, licences, access notes and loaders are in place.
Inspiration4 now has real downloaded inputs (see §4); every other real corpus
is still unpulled, and even Inspiration4 has no clinical label — nothing here
is reportable yet.
