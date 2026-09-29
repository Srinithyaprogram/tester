# Astronaut Health Monitoring — ICDM 2026 Workshop

Detects physiological/behavioural deterioration in crew members from
multimodal data. Two scripts: load a dataset, fit and score a few models.

## Quick start

```bash
pip install -r requirements.txt
python run.py                        # synthetic smoke test, ~10s on CPU
cat results/results.csv
```

Real corpora — three of them, all openly downloadable, all with a non-circular
flight-phase label:

```bash
python scripts/fetch_osdr_transcriptomics.py && python scripts/build_osdr_panels.py
python run.py --dataset genelab_astronaut     # OSD-174, 10 ISS astronauts
python run.py --dataset osdr_bedrest          # OSD-370, 10 bed-rest subjects
python run.py --dataset inspiration4          # OSD-569/575, 4 crew
```

The VLM benchmark (Tables 1–4) lives in `vlm_bench/` — start with
`vlm_bench/ASTRO_RESULTS.md`.

## Layout

- `datasets.py` — dataset loading. `synthetic_crew_gait` is generated in
  memory; every other key reads `data/raw/<key>/panel.csv`.
- `run.py` — models (three classical baselines + a numpy MLP), the eval
  loop, metrics, and the CLI. `python run.py --help` for all flags.
- `scripts/` — `fetch_osdr.py` (NASA OSDR API client), `fetch_inspiration4_files.py`
  + `build_inspiration4_panel.py` (pull and merge the real I4 panel),
  `validate_panel.py` (check a panel before you model it),
  `make_example_panel.py` (format template), `run_all.sh` (every model x
  every dataset).
- `docs/` — `OSDR_FINDINGS.md` (**read first**), `DATASETS.md` (data format
  and how to fetch it).
- `data/`, `results/` — gitignored inputs/outputs; `results/results.csv` is
  overwritten each run.

## Guardrails

- **Balanced accuracy is primary**, not accuracy — deterioration is the
  minority class.
- **Splits are subject-wise by default** — no crew member in two splits.
  `--no-subject-wise-split` exists only to show how much that would inflate
  results.
- **Circular labels are flagged, not hidden.** A panel with no `label`
  column gets one derived from baseline deviation; the run is marked with a
  `validity_warning` and a `per_subject` breakdown is written for small
  cohorts (e.g. Inspiration4, n=4) so a pooled score can't hide it.
- **`synthetic_crew_gait` is a smoke test** (tuned to ~0.80–0.88 balanced
  accuracy so breakage is visible) — never report it.
- **The movement-data framing needs a decision before real numbers matter.**
  Read `docs/OSDR_FINDINGS.md`.

## Adding a model or dataset

Add a model: a class with `fit(x, y, subject_ids)` / `predict_proba(x)` in
`run.py`, wired into `build_model()` and `MODEL_KEYS`.

Add a dataset: export a tidy `panel.csv` (`subject_id`, `timepoint`, optional
`label`, any numeric measures — see `docs/DATASETS.md`); `datasets.py` needs
no changes.
