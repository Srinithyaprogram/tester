# OSDR assay audit — does the movement data exist?

> **Update, 11 August 2026.** The audit below is still correct: no human
> movement assay exists in OSDR. But its conclusion — that Inspiration4's four
> crew were the only usable human cohort, which forced the "n is tiny" weakness
> in path A — was too pessimistic. A second pass over the assay index found
> **real astronaut cohorts with per-subject, per-timepoint processed data**:
>
> - **OSD-174** — 10 ISS astronauts, hair-follicle transcriptomics, six
>   timepoints each, balanced 20/20/20 pre/in/post-flight. Open access.
> - **OSD-370** — 10 bed-rest subjects, vastus lateralis transcriptomics,
>   before/during/after. Open access.
>
> Both are now built as `genelab_astronaut` and `osdr_bedrest`
> (`scripts/fetch_osdr_transcriptomics.py` → `scripts/build_osdr_panels.py`),
> and both ship a non-circular label taken from NASA's own deposited factor
> values. That is 2.5x the subjects of Inspiration4 with balanced classes.
>
> **This partly answers the proposal's bed-rest question too.** The proposal
> names MR079G HDT30, Campaign 17 and the UTMB campaigns — all LSDA
> managed-access. OSD-370 is an openly-downloadable bed-rest analog cohort.
> It carries transcriptomics rather than strength and functional-performance
> measures, so it substitutes for the *cohort*, not for the *modality*.
>
> Checked and rejected: **OSD-530** (6 JAXA astronauts, cell-free RNA) deposits
> only group-level means and SEM — no per-sample matrix short of running an
> RNA-seq pipeline from fastq. **OSD-483** deposits fastq only.
>
> Path A is therefore stronger than described below, and path C (an LSDA data
> request) is the only route to actual movement data. Results:
> `vlm_bench/ASTRO_RESULTS.md`.

Queried the NASA OSDR Biological Data API on **30 July 2026** to settle the open
question: do the four proposal data sources actually contain gait, posture,
balance or actigraphy measurements for human crew?

**Answer: no. Not one human movement assay exists in OSDR.**

This is not a "we couldn't find it" result. It is the complete enumeration of
assay types across every human accession in the repository.

---

## What was run

```
GET /v2/query/assays/?study.characteristics.organism=Homo%20sapiens
                     &assay.technology%20type&assay.measurement%20type
GET /v2/query/assays/?assay.measurement%20type          # all organisms
```

Base: `https://visualization.osdr.nasa.gov/biodata/api/v2`

Reproduce with `python scripts/fetch_osdr.py --search-assays gait balance sway locomotion actigraphy`.

---

## Every assay type in OSDR's human accessions

All ~121 *Homo sapiens* accessions, complete list:

| Assay measurement type | Accessions |
|---|---|
| transcription-profiling | ~96 accessions (the bulk of the archive) |
| dna-methylation-profiling | OSD-86, 140, 317, 943 |
| protein-expression-profiling | OSD-315, 353, 359, 571 |
| metabolite-profiling | OSD-571, 832, 833, 834 |
| genome-sequencing | OSD-569, 571 |
| targeted-genome-sequencing | OSD-569 |
| complete-blood-count | OSD-569 |
| chromatin-accessibility | OSD-570 |
| analytical-chemistry | OSD-575 |
| immunoassay | OSD-575, 656 |
| protein-dna-binding-site-identification | OSD-687 |
| image-analysis | OSD-815 |
| electrophysiology | OSD-816 |
| histological-assay | OSD-817 |
| molecular-cellular-imaging | OSD-817 |
| protein-quantification | OSD-818 |
| translation-profiling | OSD-936 |

Technology types are correspondingly molecular: RNA-seq, microarray, mass
spectrometry, single-cell ATAC/RNA-seq, real-time PCR, western blot,
immunohistochemistry, hematology analyzer, electroretinography, light microscopy.

**Nothing kinematic. No accelerometry, no actigraphy, no gait, no posturography,
no functional performance test.** The closest thing to a physical measure in the
human data is a complete blood count and a metabolic panel.

## Movement data does exist in OSDR — for animals

Querying across all organisms returns these behavioural assays:

```
behavior_gait_digigait-system-(mouse-specifics)
behavior_balance_conduct science (me-bb-m01)
behavior_locomotory-behavior_manual counting-1
behavior_open-field_ethovisionxt_12
behavior_elevated-plus-maze_ethovisionxt_12
behavior_novel-object-recognition_ethovisionxt_12
behavior_rawm_ethovisionxt_12
behavior_three-chamber-social-isolation_ethovisionxt_12
behavior_video-recording_multi-use variable gravity drosophila hardware
```

Mouse and *Drosophila*. Open access, no application required. Registered in this
repo as `osdr_rodent_behaviour`.

## Where human movement data actually lives

Not OSDR. NASA's **Life Sciences Data Archive (LSDA)**, served through the NASA
Life Sciences Portal:

- https://nlsp.nasa.gov/explore/lsdahome
- https://lsda.jsc.nasa.gov/

LSDA covers "Neuroscience", "Behavior and performance", "Cardiovascular",
"Muscle physiology" — 30 years of astronaut data from the Human Research Program.
That is where sensorimotor, balance and functional-performance measures would be.

**But:** it operates on a **managed-access model**. There is a formal *Data
Request* process, and no public API or bulk download. Getting human movement data
means submitting a request and waiting — weeks, not an afternoon.

---

## What this means for the paper

The movement framing cannot be executed on openly-downloadable human data. Three
honest paths, in order of how fast you can start:

**A. Reframe to multimodal physiological deterioration — human, start today.**
Use OSDR / Inspiration4 omics and clinical panels. Deterioration is defined
against each crew member's own preflight baseline. The infrastructure already
does this unchanged; only the framing text moves. Weakness: n is tiny (I4 is 4
crew) and the modality is molecular, so "behavioural health monitoring" has to
come out of the title.

**B. Rodent gait and balance — movement, start today.**
`osdr_rodent_behaviour`. Real gait kinematics (DigiGait), real balance assays,
open access, larger n than any human cohort. A defensible ICDM paper:
deterioration detection from spaceflight-induced gait change. Weakness: it is not
astronaut health monitoring, and you would need to argue translational relevance
rather than assert it.

**C. Human movement via LSDA — the original vision, slower.**
Submit a data request to LSDA for sensorimotor / behaviour-and-performance data.
Correct for the proposal as written, but the timeline is a request cycle, and the
ICDM workshop deadline may not accommodate it. Worth submitting the request now
in parallel with A or B regardless — it costs little and unblocks later work.

**A hybrid is probably strongest:** run B for the movement result (real
kinematics, adequate n), and A as a human-cohort transfer arm, with the honest
framing that the human validation is molecular and the movement evidence is
preclinical. That is a complete story with data you can download today.

---

## One caveat on this audit

This enumerates what the OSDR *biodata API* indexes. It is possible ALSDA holds
human physiological data that is catalogued in LSDA but not exposed through this
API — the two systems are distinct, and ALSDA's non-omics holdings appear to
surface through LSDA rather than OSDR. The audit is conclusive about OSDR; it is
not proof that no NASA human movement data exists anywhere. It does establish
that none of it is openly downloadable through the API the proposal points at.

Sources: [OSDR Biological Data API](https://visualization.osdr.nasa.gov/biodata/api/),
[NASA Life Sciences Portal](https://nlsp.nasa.gov/explore/lsdahome),
[LSDA](https://lsda.jsc.nasa.gov/),
[ALSDA](https://www.nasa.gov/ames/space-biosciences/ames-life-sciences-data-archive/)
