# Start here

Experiment infrastructure and results for an ML paper on astronaut health
monitoring, targeting the ICDM 2026 Workshop.

**Three real human corpora, real labels, and Tables 1–4 populated** — see
`vlm_bench/ASTRO_RESULTS.md` for the numbers and what they do and don't
support. `docs/OSDR_FINDINGS.md` explains why the movement framing in the
original proposal cannot be executed on open data, and what replaced it.

**Eight models × three real corpora**, all encoded and scored, with
subject-level confidence intervals, paired significance tests, confusion
counts, and a stricter-label robustness pass. No table cell is a placeholder.

Headline findings, all negative and all clean:

- **Zero-shot VLM prompting fails.** 16 of 24 model×corpus configurations are
  constant classifiers (balanced accuracy 0.500, MCC 0); 12 of 24 rank the
  classes backwards.
- **Biomedical pretraining does not help.** One corpus significantly favours
  general-purpose (p=0.009), two are null. Nothing favours biomedical — the
  opposite of the proposal's H1.
- **Cross-corpus transfer fails.** All four Table 3 rows at balanced accuracy
  0.479–0.571; BLIP-2's is worse than guessing.
- **No VLM beats the raw measurements** (wash on two corpora, significantly
  worse on bed rest, p=0.008), and multimodal fusion loses to physiology
  alone (RQ4).

**And the constraint on all of it: no ranking here is stable.** Changing the
model pool or switching to the stricter label flips the winner on two of three
corpora. Report the qualitative findings, not a leaderboard. Bed rest is a null
result outright — no model on it beats chance.

The large models were encoded on `ssh amdcluster` (AMD MI300X, ROCm) via
`vlm_bench/encode_cluster.py`; CLIP and BiomedCLIP run on a laptop.

## 1. Check it works (1 minute)

```bash
pip install -r requirements.txt
python run.py
pytest tests/ -q
```

Expect balanced accuracy roughly 0.50–0.89 across five models on a simulated
corpus. If everything reads 1.000, something is broken — the simulator is
deliberately imperfect so pipeline breakage is visible.

## 2. The finding: the movement data isn't where the proposal says it is

Audited NASA's OSDR API on 30 July 2026. Across all ~121 human accessions,
**no gait, posture, balance, actigraphy or accelerometry assay exists** —
every human assay is molecular (transcription, methylation, protein/metabolite
profiling) plus a blood count. Movement data *does* exist in OSDR, but only
for mouse and Drosophila. Human sensorimotor data lives in NASA's separate
LSDA archive, which is managed-access (a data request, no public API).

Three honest paths (full detail in `docs/OSDR_FINDINGS.md`):

- **A.** Reframe to human molecular/physiological deterioration. Start today.
- **B.** Rodent gait/balance data. Real kinematics, open access. Start today.
- **C.** Request human movement data from LSDA. Correct as proposed, but a
  request cycle may not fit the deadline — submit in parallel regardless.

A hybrid of B (movement result) + A (human transfer arm) is a complete story
using data downloadable today. **This is a science decision — settle it
before writing the abstract.**

## 3. What's real and what isn't

| | Status |
|---|---|
| The code (`datasets.py`, `run.py`) | Real. Runs, tested. |
| The OSDR audit | Real. Queried NASA's live API (re-verified, not just cited). |
| `synthetic_crew_gait` | Simulated. A smoke test. **Never report it.** |
| `data/raw/inspiration4/panel.csv` | **Real** — CBC, metabolic panel, immune/cardiovascular multiplex (OSD-569/575). 4 crew, 28 samples. Now carries a real flight-phase label. |
| `data/raw/genelab_astronaut/panel.csv` | **Real** — OSD-174, 10 ISS astronauts, hair-follicle microarray, 60 samples, balanced pre/in/post-flight. |
| `data/raw/osdr_bedrest/panel.csv` | **Real** — OSD-370, 10 bed-rest subjects, muscle microarray, 60 samples. |
| `alsda_physiology` | Fabricated (`make_example_panel.py`). Now carries a `SYNTHETIC.txt` marker and `validate_panel.py` hard-errors on it. Superseded by `osdr_bedrest`. |
| `osdr_human_spaceflight`, `genelab_omics`, `osdr_rodent_behaviour` | Nothing pulled — `missing_data`. |
| `results/results.csv` | Still the simulator. Not a result. |
| `vlm_bench/results/*.csv` | **Real.** Tables 1–4 on the three corpora above — see `vlm_bench/ASTRO_RESULTS.md`. |
| `descriptor_mlp` | Real, tested, torch-free — the one deep row that runs anywhere. |

### The label question is settled

All three real corpora now carry a **flight-phase label taken from NASA's own
deposited factor values** — mission metadata, independent of the measurements.
That removes the circularity that made every earlier number unreportable. It is
an exposure proxy, not a clinical diagnosis; report it as such.

All three tasks are learnable from the raw measurements under
leave-one-subject-out (AUROC 0.78 / 0.82 / 0.88, `vlm_bench/reference_baselines.py`),
so a VLM row near chance is a fact about the model, not an impossible task.

### One trap worth knowing before you touch the rendering

Z-scoring each sample against its own crew member's preflight draws leaks the
label — preflight deviation is pinned near zero by arithmetic — and produced a
fake CLIP AUROC of 0.995. The default rendering is now cohort-level, and a
`deviation_magnitude` control row plus a per-manifest `magnitude_control_auroc`
exist to catch any recurrence. Full detail in `vlm_bench/ASTRO_RESULTS.md`.

## 4. To add data

Read `docs/DATASETS.md`. Short version: one tidy CSV per corpus at
`data/raw/<dataset_key>/panel.csv` with `subject_id`, `timepoint`, ideally a
`label`, and any numeric measures.

```bash
python scripts/make_example_panel.py --dataset inspiration4   # see the layout
python scripts/validate_panel.py --dataset inspiration4       # check yours
```

**Inspiration4 is a partial exception** — real data, rebuildable any time:

```bash
python scripts/fetch_osdr.py --inspiration4        # cache the manifest
python scripts/fetch_inspiration4_files.py         # pull the actual files (~330MB)
python scripts/build_inspiration4_panel.py         # merge into panel.csv
```

Deliberately scoped to the 5 accessions with small, clinically-usable files
(CBC, metabolic panel, proteomics/metabolomics, immune/cardiovascular
multiplex). Excluded: raw sequencing (fastq/bam, easily terabytes across the
full I4 accession set) and the metagenomic-sequencing accessions' per-sample
pipeline output tables (thousands of files, need a separate bioinformatics
workflow to become per-subject features) — not usable by this repo's
tidy-panel pipeline as-is.

`build_inspiration4_panel.py` now also writes `phase`, `label` and
`label_acute` from the mission timeline, so this corpus no longer triggers the
circular baseline-deviation fallback.

The two transcriptomic corpora come from a separate pair of scripts:

```bash
python scripts/fetch_osdr_transcriptomics.py   # ~80MB: OSD-174 + OSD-370
python scripts/build_osdr_panels.py            # -> genelab_astronaut, osdr_bedrest
```

The remaining gap is no longer the label — it is that the label is **flight
phase, not a clinical diagnosis**. Nothing here distinguishes a deteriorated
crew member from a merely spaceflight-exposed one. Closing that needs clinical
adjudication that no open NASA dataset provides.

The validator catches what otherwise wastes a session: missing columns, no
preflight baseline, subjects leaking across splits, constant/all-NaN columns,
single-class labels, cohorts too small to split, and panels marked
`SYNTHETIC.txt`.

Dataset keys: `osdr_human_spaceflight`, `genelab_omics`, `alsda_physiology`,
`inspiration4`, `genelab_astronaut`, `osdr_bedrest`, `osdr_rodent_behaviour`.
Missing corpora are reported as `missing_data`, not a crash.

## 5. Five traps, already guarded — don't disable the guards

**Circular labels.** No `label` column → one is derived by thresholding
baseline deviation, flagged with a `validity_warning` in `results.csv`. That
label is a function of the model's own input, so 1.000 means nothing. The three
real corpora now ship a real label and never hit this path.

**Leaky renderings.** Z-scoring a sample against its own crew member's
preflight draws pins preflight deviation near zero by arithmetic and hands any
model a free 0.98+ AUROC. `vlm_bench/astro_modalities.py` defaults to
cohort-level z, records `magnitude_control_auroc` in every manifest, and warns
above 0.9. `reference_baselines.py` carries a permanent `deviation_magnitude`
row. Whenever a VLM result looks too good, check that row first.

**Fabricated panels.** `make_example_panel.py` output is random numbers. It now
writes `SYNTHETIC.txt` beside every panel it generates and `validate_panel.py`
treats that as a hard error. Don't delete a marker to make a run go through.

**Missingness as a shortcut.** Inspiration4 is missing one crew member's entire
406-analyte immune panel at one preflight draw; median imputation turns that
into a label signal worth ~+0.10 balanced accuracy. `validate_panel.py` now
warns when the per-row NaN count predicts the label.

**Study-design covariates.** OSD-370's recovery arm was only assayed under one
treatment condition, so "is this insulin-stimulated" partly predicts the bed-rest
label (AUROC 0.603; `label_acute` drops it to 0.521). The validator now warns
when any low-cardinality covariate tracks the label.

**Leaky splits.** Subject-wise by default. `--no-subject-wise-split` exists
to show the inflation (~0.81 → 0.97 on simulated data) — don't cite it.

**Tiny cohorts.** Inspiration4 flew four people. Use the `per_subject` column
in `results.csv`, not a pooled accuracy. Expect deep models to overfit —
that's a finding.

Balanced accuracy is primary, not accuracy — deterioration is the minority
class.

---

Full detail: `README.md`, `docs/OSDR_FINDINGS.md`, `docs/DATASETS.md`.
