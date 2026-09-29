# Tables 1–4 on astronaut data — as actually run

`RESULTS.md` answers RQ2 on public medical imaging benchmarks, because at the
time no astronaut corpus in this repo had both usable inputs and a defensible
label. That has changed. Three real, openly-downloadable human corpora now
carry a non-circular label, so RQ1, RQ2, RQ3 and RQ4 can be run on astronaut
data directly. This file is those results.

Everything below is real inference on real NASA data. Nothing is simulated,
and no cell is left to be filled in later — **eight models × three corpora**,
all encoded and scored, with subject-level confidence intervals, paired
significance tests, confusion counts, and a stricter-label robustness pass.

**Four findings, all negative:**

1. **Zero-shot prompting fails.** 16 of 24 model×corpus configurations are
   constant classifiers (balanced accuracy exactly 0.500, MCC exactly 0), and
   12 of 24 rank the classes backwards (AUROC < 0.5).
2. **Biomedical pretraining does not help.** One corpus significantly favours
   general-purpose VLMs (p=0.009); the other two are null. Nothing favours
   biomedical.
3. **Cross-corpus transfer fails.** All four Table 3 rows sit at balanced
   accuracy 0.479–0.571; one is worse than guessing.
4. **No VLM beats the raw measurements.** Two corpora are a wash, and on bed
   rest the raw features win significantly (p=0.008). Multimodal fusion loses
   to physiology alone (RQ4).

**And one methodological finding that constrains all of the above:** no ranking
in this benchmark is stable. Changing which models represent a family, or
switching to the stricter label, changes the winner on two of three corpora.
Report the qualitative conclusions; do not report a leaderboard.

Taken together these argue *against* the multimodal framing for tabular
astronaut health data. That is a legitimate benchmark contribution — a
reference point saying where the ceiling currently is — but it is not the
result the proposal's H1 predicted, and it should not be written up as if it
were.

## The corpora

| Key | Source | Subjects | Samples | Measurements | Label (from NASA's own factor values) |
|---|---|---|---|---|---|
| `inspiration4` | OSD-569 / OSD-575 | 4 civilian crew | 28 | 605 clinical lab values | pre-flight (12) vs post-flight (16) |
| `genelab_astronaut` | OSD-174 | 10 ISS astronauts | 60 | 1024 microarray probes (hair follicle) | pre-flight (20) vs in/post-flight (40) |
| `osdr_bedrest` | OSD-370 | 10 bed-rest subjects | 60 | 1024 microarray probes (vastus lateralis) | before (25) vs during/after bed rest (35) |

The label is the **flight phase from the mission timeline**, taken verbatim
from the deposited `Factor Value[Spaceflight]` / `Factor Value[Bed Rest]`. It is
an exposure proxy, not a clinical deterioration diagnosis — say so wherever it
is reported. Critically, it is *not* a function of the measurements, so it does
not have the circularity of the baseline-deviation fallback in `datasets.py`.

Rebuild any of it:

```bash
python scripts/fetch_osdr_transcriptomics.py
python scripts/build_osdr_panels.py
python scripts/build_inspiration4_panel.py
```

## Read every table against these reference points

A zero-shot VLM number in isolation says nothing. `reference_baselines.py`
gives each corpus a floor, a ceiling, and a control (leave-one-subject-out,
balanced accuracy):

| Corpus | majority floor | deviation-magnitude control | best raw-feature ceiling |
|---|---|---|---|
| Inspiration4 | 0.500 | 0.646 | 0.771 (logistic regression) |
| GeneLab OSD-174 | 0.500 | 0.725 | 0.738 (gradient boosting) |
| Bed rest OSD-370 | 0.500 | 0.517 | 0.794 (logistic regression) |

All three tasks are learnable from the raw measurements (AUROC 0.78 / 0.82 /
0.88). That matters: it means a VLM row near chance is a fact about the model,
not about an impossible task.

**The deviation-magnitude control is not decoration.** It is a single scalar —
the mean absolute z-score of a sample, no model, no fitting. See "The rendering
bug that had to be fixed" below.

## Table 1 (RQ1) — general-purpose VLMs

All four rows are real inference. Zero-shot and linear probe are both reported
because they disagree sharply, and that disagreement is the finding.

| Model | Dataset | Protocol | Acc | Bal.Acc | Prec | Rec | F1 | AUROC | MCC |
|---|---|---|---|---|---|---|---|---|---|
| CLIP | Inspiration4 | zero-shot | 0.643 | 0.604 | 0.636 | 0.875 | 0.737 | 0.677 | 0.251 |
| CLIP | Inspiration4 | linear probe | 0.786 | 0.781 | 0.812 | 0.812 | 0.812 | 0.844 | 0.562 |
| BLIP-2 | Inspiration4 | zero-shot | 0.571 | 0.500 | 0.571 | 1.000 | 0.727 | 0.349 | 0.000 |
| BLIP-2 | Inspiration4 | linear probe | 0.750 | 0.750 | 0.800 | 0.750 | 0.774 | 0.776 | 0.496 |
| LLaVA-1.6 | GeneLab OSD-174 | zero-shot | 0.667 | 0.500 | 0.667 | 1.000 | 0.800 | 0.791 | 0.000 |
| LLaVA-1.6 | GeneLab OSD-174 | linear probe | 0.767 | 0.713 | 0.795 | 0.875 | 0.833 | 0.836 | 0.453 |
| Qwen2.5-VL | GeneLab OSD-174 | zero-shot | 0.333 | 0.500 | 0.000 | 0.000 | 0.000 | 0.664 | 0.000 |
| Qwen2.5-VL | GeneLab OSD-174 | linear probe | 0.767 | 0.738 | 0.825 | 0.825 | 0.825 | 0.811 | 0.475 |

**Three of the four collapse to a single class at zero-shot** — balanced
accuracy exactly 0.500, MCC exactly 0. BLIP-2 and LLaVA-1.6 answer "yes" to
every sample (recall 1.000); Qwen2.5-VL answers "no" to every sample (recall
0.000). Only CLIP produces a non-degenerate zero-shot decision, and it is weak.

**The representations are not the problem — the prompting is.** LLaVA-1.6 has
zero-shot AUROC 0.791 while its balanced accuracy sits at 0.500: it ranks
exposed samples above baseline ones almost as well as its own linear probe
does, but no threshold on its yes/no logit expresses that ranking. Qwen2.5-VL
shows the same split (AUROC 0.664, balanced accuracy 0.500).

Under a linear probe all four land at 0.713–0.781 balanced accuracy, at or
slightly above the raw-feature ceilings of 0.771 (Inspiration4) and 0.738
(GeneLab). The signal is in the frozen embeddings; the instruction-following
path cannot reach it.

Accuracy is the misleading column here. LLaVA-1.6 zero-shot reads 0.667
accuracy and 0.800 F1 while being a constant classifier. Read balanced accuracy
and MCC.

## Table 2 (RQ2) on astronaut data — biomedical vs general-purpose

`RESULTS.md` could only answer RQ2 by proxy, on chest X-rays. This is the
matched comparison: **all eight models, all three corpora**, same rendering,
same protocol — four general-purpose VLMs against the four biomedical ones the
proposal names. 24 configurations.

Linear-probe balanced accuracy / MCC (full table with every metric and
confusion counts in `results/table2_astro.csv`):

| Model | Family | Inspiration4 | GeneLab OSD-174 | Bed rest OSD-370 |
|---|---|---|---|---|
| CLIP | general | 0.781 / 0.562 | 0.762 / 0.572 | 0.391 / −0.222 |
| BLIP-2 | general | 0.750 / 0.496 | **0.800** / 0.617 | 0.463 / −0.081 |
| LLaVA-1.6 | general | **0.844** / **0.710** | 0.713 / 0.453 | 0.543 / 0.089 |
| Qwen2.5-VL | general | 0.698 / 0.409 | 0.738 / 0.475 | 0.591 / 0.191 |
| BiomedCLIP | biomedical | 0.594 / 0.190 | **0.800** / **0.653** | 0.563 / 0.129 |
| Med-Flamingo | biomedical | 0.688 / 0.411 | 0.750 / 0.507 | 0.543 / 0.089 |
| CheXagent | biomedical | 0.635 / 0.271 | 0.787 / 0.583 | **0.606** / **0.224** |
| BioViL-T | biomedical | 0.542 / 0.091 | 0.488 / −0.047 | 0.523 / 0.048 |

**The RQ2 hypothesis is not supported.** Best general-purpose vs best
biomedical, per corpus, with the paired subject bootstrap:

| Corpus | Best general | Best biomedical | Δ bal.acc | p |
|---|---|---|---|---|
| Inspiration4 | LLaVA-1.6 0.844 | Med-Flamingo 0.688 | +0.156 | **0.009** |
| GeneLab OSD-174 | BLIP-2 0.800 | BiomedCLIP 0.800 | 0.000 | 1.000 |
| Bed rest OSD-370 | Qwen2.5-VL 0.591 | CheXagent 0.606 | −0.014 | 0.940 |

One corpus significantly favours general-purpose; two are null. **Nothing
favours biomedical at any conventional threshold.**

This reverses the earlier reading in this file, which compared only CLIP
against BiomedCLIP and concluded biomedical won 2 of 3. Widening the pool
inverted it — and adding CheXagent and BioViL-T then flipped the *bed rest*
winner back to biomedical, on a corpus where nothing is significant anyway.
Three different model pools, three different answers.

**The spread within the biomedical family is larger than the gap between
families.** CheXagent (0.787 on GeneLab) and BioViL-T (0.488, MCC −0.047) are
both "biomedical VLMs" and sit at opposite ends of the table. Any claim of the
form "biomedical models do X" is really a claim about a specific checkpoint.

The honest interpretation: these heatmaps are out-of-distribution for *both*
families. Nothing in any of these models' pretraining includes a false-colour
grid of z-scored microarray probes, so the biomedical prior buys nothing and
what remains is generic texture encoding.

That reading is supported directly by the embeddings. Mean pairwise cosine
similarity between distinct samples' image embeddings is **0.985 for BioViL-T
and 0.94–0.96 for BiomedCLIP** — the encoders map every heatmap to almost the
same point. There is very little for a probe to work with, which is why the
probe results are modest and why BioViL-T, the most concentrated of them, is
the weakest.

### Zero-shot prompting fails almost universally

**16 of 24 zero-shot configurations are constant classifiers** — every sample
assigned to one class, balanced accuracy exactly 0.500, MCC exactly 0. The
confusion counts now carried on every row make this checkable at a glance: an
entire column of zeros.

**12 of 24 rank the classes backwards** (zero-shot AUROC < 0.5) — exactly half.
BLIP-2 on Inspiration4 reaches 0.349, meaning its yes/no logit is
anti-correlated with flight phase.

The ranking is not always the problem. LLaVA-1.6 on GeneLab has zero-shot AUROC
0.791 — nearly its own linear probe's 0.836 — while sitting at balanced accuracy
0.500. The information is there in the logit ordering; no threshold on that
logit recovers it. Prompting is the failure point, not perception.

### One near-miss worth recording

CheXagent's first run returned zero-shot AUROC of **exactly 0.500 on all three
corpora**. That was not a result: `AutoProcessor` for CheXagent-2-3b returns a
bare tokenizer with `image_processor = None`, so `images=[...]` was silently
dropped with a "Keyword arguments not recognized" warning and the model scored
an identical text prompt every time. Images go in as `<|img|>path<|/img|>`
tokens instead. `encode_cluster.py` now raises if a model returns an identical
score for every sample, so this specific way of manufacturing a fake row cannot
recur silently.

## Table 3 (RQ3) — cross-dataset generalisation

Fit the probe on every sample of one corpus, predict every sample of another.
The cohorts are disjoint by construction, so no subject can leak.

| Model | Train | Test | Acc | Bal.Acc | Prec | Rec | F1 | AUROC | MCC |
|---|---|---|---|---|---|---|---|---|---|
| CLIP | Inspiration4 | GeneLab OSD-174 | 0.650 | 0.550 | 0.694 | 0.850 | 0.764 | 0.568 | 0.122 |
| BLIP-2 | GeneLab OSD-174 | Inspiration4 | 0.536 | 0.479 | 0.560 | 0.875 | 0.683 | 0.609 | −0.067 |
| BiomedCLIP | Inspiration4 | Bed rest OSD-370 | 0.633 | 0.571 | 0.623 | 0.943 | 0.750 | 0.530 | 0.219 |
| Med-Flamingo | GeneLab OSD-174 | Bed rest OSD-370 | 0.483 | 0.529 | 0.643 | 0.257 | 0.367 | 0.562 | 0.067 |

**Transfer does not happen.** All four rows land at balanced accuracy
0.479–0.571 and MCC −0.067 to 0.219 — at or below chance, and far below the
same models' within-corpus probes (0.750–0.844 on their training corpora).
BLIP-2's transfer MCC is *negative*: worse than guessing.

The failure mode is visible in the precision/recall split. Three of four rows
pair high recall (0.850–0.943) with near-chance AUROC — the classifier has
collapsed toward calling almost everything "exposed". Med-Flamingo fails the
other way, recall 0.257, calling almost everything baseline. Neither is
discrimination.

A decision boundary learned on 4 crew members' blood chemistry does not survive
the move to 10 astronauts' hair-follicle transcriptomes. The corpora share a
label *definition*, not a feature space: the VLM embedding was supposed to be
the common representation that made the transfer well-posed, and it is not.
This is the strongest negative result in the set — it holds across both model
families and every corpus pairing tried.

Note the substitution: the proposal's Table 3 names "NASA ALSDA" as the test
corpus. No ALSDA export exists here — the file that carried that name was
fabricated (see below) — so OSD-370 stands in as the real analog cohort.

## Table 4 (RQ4) — modality ablation

BiomedCLIP on Inspiration4, leave-one-subject-out, same classifier throughout.
All three modalities are derived from the same row, so this compares
representations of identical information.

| Configuration | Features | Accuracy | Precision | Recall | F1 | AUROC | MCC |
|---|---|---|---|---|---|---|---|
| Image only | 512 | 0.607 | 0.647 | 0.688 | 0.667 | 0.651 | 0.190 |
| Text only | 512 | 0.714 | 0.722 | 0.812 | 0.765 | 0.802 | 0.409 |
| Physiological data only | 605 | **0.786** | 0.778 | 0.875 | **0.824** | 0.776 | **0.559** |
| Image + text + physiological | 1629 | 0.750 | 0.765 | 0.812 | 0.788 | **0.812** | 0.486 |

**The ordering is physio > text > image, and fusion does not win.** The raw
measurements beat every learned representation of them (MCC 0.559 vs 0.409 vs
0.190), and concatenating all three is *worse* than physiology alone on every
metric except AUROC. The vision tower is the weakest link: rendering a
biomarker table as a heatmap and encoding the picture destroys information that
was already in the numbers.

This is the expected outcome and it is worth stating plainly — it argues
against the multimodal framing for tabular astronaut data, not for it.

## How much of this is noise? (confidence intervals and significance)

`vlm_bench/astro_stats.py` bootstraps at the **subject** level — resampling
whole crew members, not samples, because samples from one person are repeated
measures and bootstrapping over them would give intervals several times too
narrow. Model-vs-model tests are paired on the same resampled cohort.

95% CIs on linear-probe balanced accuracy are wide enough to change how the
tables should be read:

| Corpus | Best model | Point | 95% CI |
|---|---|---|---|
| Inspiration4 (k=4) | LLaVA-1.6 | 0.844 | [0.719, 0.958] |
| GeneLab (k=10) | BLIP-2 | 0.800 | [0.688, 0.900] |
| Bed rest (k=10) | CheXagent | 0.606 | [0.463, 0.749] |

**No model on bed rest is significantly better than chance.** Seven of the
eight MCC intervals contain zero; the eighth, CLIP's, excludes zero on the
*negative* side ([−0.451, −0.004]) — significantly worse than guessing. Bed
rest should be reported as a null result, not ranked.

Paired tests on the comparisons the write-up actually makes
(`results/stats_comparisons.csv`):

| Comparison | Corpus | Δ balanced acc | p |
|---|---|---|---|
| best general − best biomedical | Inspiration4 | +0.156 | **0.009** |
| best general − best biomedical | GeneLab | 0.000 | 1.000 |
| best general − best biomedical | Bed rest | −0.014 | 0.940 |
| linear probe − zero shot | Inspiration4 | +0.06 to +0.34 | <0.001 (4/6) |
| linear probe − zero shot | GeneLab | +0.21 to +0.31 | <0.001 (6/6) |
| linear probe − zero shot | Bed rest | −0.11 to +0.09 | ns (5/6) |
| best VLM probe − raw-feature ceiling | Inspiration4 | +0.073 | 0.560 |
| best VLM probe − raw-feature ceiling | GeneLab | +0.062 | 0.386 |
| best VLM probe − raw-feature ceiling | Bed rest | −0.189 | **0.008** |

What survives testing:

- **The probe-beats-prompting result is real** — significant on 10 of 18
  model×corpus pairs and never significantly reversed except CLIP on bed rest.
- **No VLM beats the raw measurements anywhere.** Two corpora are a wash, and
  on bed rest the raw features win significantly. (The ceiling is the stronger
  of logistic regression and gradient boosting per corpus — testing against
  logistic regression alone would have manufactured a spurious "BLIP-2 beats
  the raw features" on GeneLab at p=0.001.)
- **RQ2 favours general-purpose on one corpus and is null on two.** Nothing
  favours biomedical at any conventional threshold.

Two caveats that cut against over-reading even this: "best of family" is chosen
on the same data it is then tested on, which inflates the one significant RQ2
result; and with k=4 subjects the bootstrap cannot resolve p below roughly
0.005 no matter how large the true effect.

## Does it survive a stricter label? (robustness)

`label` follows NASA's factor value, which lumps recovery timepoints in with
the exposed class — Inspiration4's R+82/R+194 draws are months after return.
`label_acute` drops them. `vlm_bench/astro_robustness.py` re-scores everything
on that subset (28→20, 60→40, 60→48 samples).

**The conclusions hold; the rankings do not.**

| | `label` | `label_acute` |
|---|---|---|
| Best on Inspiration4 | LLaVA-1.6 0.844 | BLIP-2 0.812 |
| Best on GeneLab | BLIP-2 / BiomedCLIP 0.800 | BiomedCLIP 0.875 |
| Best on bed rest | CheXagent 0.606 | LLaVA-1.6 0.584 |

Individual models move by up to ±0.15 balanced accuracy, in both directions,
and **the winner changes on two of three corpora**. Under `label_acute`
BiomedCLIP wins GeneLab outright (0.875 / MCC 0.751) where it merely tied
before — so even the direction of the RQ2 answer is label-dependent.

The raw-feature ceiling moves too, and hard: Inspiration4 falls from 0.771 to
0.583 balanced accuracy once the late post-flight draws are removed. That is
worth pausing on. It means a good part of what looked like "detecting
spaceflight exposure" in the headline Inspiration4 numbers was separating
samples drawn months apart — plausibly assay batch or storage drift rather than
physiology. The four-crew corpus cannot distinguish those.

CheXagent is the sharpest example: on Inspiration4 it moves from 0.635 to
0.792 balanced accuracy (MCC 0.271 to 0.583) purely from dropping eight rows.

Together with the model-pool sensitivity noted under Table 2, that is three
independent demonstrations that **no ranking in this benchmark is stable**.
Report the qualitative findings (prompting fails, transfer fails, VLMs do not
beat the numbers); do not report a leaderboard.

Method note: the robustness pass subsets the existing encodings rather than
re-rendering and re-encoding, so the heatmaps were normalised over the full
cohort. The model saw an identical image in both conditions, which is what
makes the difference attributable to the label.

## Verification: are these numbers right?

Three checks, all run against the published CSVs rather than asserted.

**1. Arithmetic.** Every linear-probe metric was recomputed from the `.npz`
files through an independent code path — raw sklearn, none of this repo's
helpers. 120 values across 24 model×corpus probes. Maximum disagreement
**8.3 × 10⁻¹⁷** (floating-point noise). The tables are what the encodings say.

**2. Normalisation leak, physio path: none.** The manifest's `physio` vectors
are z-scored cohort-wide, then the probe's `StandardScaler` re-standardises
per fold. Those compose to an affine map, so the per-fold scaler exactly
cancels the global normalisation — verified numerically to 1.8 × 10⁻¹⁴. Table
4's physio row and every `reference_baselines.py` number are strictly
inductive.

**3. Normalisation leak, image path: real but bounded.** Here the global
z-score is baked into pixels before a frozen encoder sees them, so the scaler
cannot undo it, and the cohort mean/std include the held-out subject. Holding
one subject out shifts z by 0.18 units on Inspiration4 (4 subjects) and ~0.06
on the ten-subject corpora, against a ±3 colour range.

Measured end-to-end on the worst case — BiomedCLIP × Inspiration4, re-rendered
and re-encoded per fold with train-only statistics:

| | Bal.acc | MCC | AUROC |
|---|---|---|---|
| as published (cohort-wide z) | 0.594 | 0.190 | 0.651 |
| strictly inductive (fold-wise z) | 0.573 | 0.167 | 0.609 |
| difference | −0.021 | −0.023 | −0.042 |

**The transductive rendering inflates results by about 0.02 balanced accuracy
on the four-subject corpus, and less on the others** — roughly a tenth of the
width of the confidence intervals, and far below any effect the write-up
claims. It is disclosed rather than removed: eliminating it would mean
re-rendering and re-encoding every fold of every model, ~200 GPU encoding runs,
to move numbers by less than the noise floor.

## Two data defects found in audit — read before citing a corpus

Neither changes the qualitative conclusions. Both change how much weight
individual numbers can carry, and both are now caught automatically by
`scripts/validate_panel.py`.

### Inspiration4: missingness is a shortcut

One crew member (C003) is missing the **entire 406-analyte Alamar immune panel**
at a single preflight draw (L-44). Every other row has at most 2 missing values.
Median imputation then makes that row distinctive, and the per-row NaN count
alone separates the classes at AUROC 0.659.

It is one row out of 28, but on a four-subject corpus that is not negligible:

| Inspiration4 raw-feature ceiling | Balanced acc | MCC | AUROC |
|---|---|---|---|
| all 605 features (as reported in the tables) | 0.771 | 0.559 | 0.781 |
| complete-case features only (194) | 0.667 | 0.471 | 0.740 |
| drop the Alamar panel entirely (199) | 0.667 | 0.471 | 0.755 |

So roughly **+0.10 balanced accuracy of the headline Inspiration4 ceiling is a
missing-data artifact**, not physiology. The corpus still clears the 0.500
floor without it, so the finding "this task is learnable" survives — but the
0.771 figure should not be quoted as the ceiling without this caveat.

Worth noting alongside: 406 of 605 Inspiration4 features (67%) come from that
one immune panel, so the corpus's feature space is dominated by a single assay.

### Bed rest: a study-design covariate tracks the label

OSD-370 assays each subject under two conditions, baseline and
insulin-stimulated. The **"after bed rest and retraining" arm was only ever
assayed at baseline** (12 baseline, 0 insulin-stimulated), so treatment
condition partly predicts the label:

| Label | treatment → label AUROC |
|---|---|
| `label` (Before=0; During+After=1) | 0.603 |
| `label_acute` (After dropped) | 0.521 |

A model can score partly by detecting "is this insulin-stimulated" rather than
"is this subject deconditioned". `label_acute` removes it almost entirely, which
is an independent reason to prefer that label on this corpus. Bed rest is a null
result either way, so nothing downstream depends on it.

### What the validator now catches

Both of these were found by hand and neither is visible in any per-column
check, so `validate_panel.py` gained two guards: one warns when the per-row NaN
count predicts the label, one warns when any low-cardinality covariate does.
They fire on exactly these two corpora and stay silent on `genelab_astronaut`,
which is clean on both counts.

## The rendering bug that had to be fixed

The first version of this benchmark z-scored every sample against **that crew
member's own preflight draws**. It produced CLIP zero-shot AUROC 0.995 on
Inspiration4.

That number was an artifact. A preflight sample is one of the handful of
samples used to *compute* its own baseline, so its deviation is pinned near
zero by arithmetic, while a postflight sample has no such bound. Preflight
heatmaps rendered nearly flat and postflight ones rendered noisy. The task had
collapsed into "is this image colourful".

Measured, not assumed — under that rendering a single scalar, the mean absolute
z-score with no model at all, separated the classes at:

| | Inspiration4 | GeneLab | Bed rest |
|---|---|---|---|
| subject-baseline rendering | 0.984 | 1.000 | 1.000 |
| cohort rendering (current default) | 0.568 | 0.764 | 0.570 |

The fix was to z-score across the whole corpus instead, so both classes sit on
the same footing and a model must identify *which* measurements moved rather
than how much anything moved. Three guards now keep this from recurring:

- `cohort_z` is the default rendering; `baseline_z` prints a warning and is
  kept for inspection only.
- Every manifest records `magnitude_control_auroc`, and rendering prints a
  loud warning above 0.9.
- `deviation_magnitude` is a permanent row in `reference_baselines.py`.

Leave-one-out baselining was tried first and is not sufficient: OSD-174 gives
each subject only two preflight draws, so removing one leaves no usable
standard deviation.

## Fabricated data, quarantined

`data/raw/alsda_physiology/panel.csv` is generated by
`scripts/make_example_panel.py` — every value comes from a random number
generator. It was previously indistinguishable from a real export once written
to disk. Now `make_example_panel.py` drops a `SYNTHETIC.txt` marker beside
every panel it generates, `validate_panel.py` treats that marker as a hard
error, and the marker has been backfilled onto the existing directory. It
appears in no table here.

## Reproducing

```bash
python scripts/fetch_osdr_transcriptomics.py     # ~80MB from OSDR
python scripts/build_osdr_panels.py
python scripts/build_inspiration4_panel.py
python vlm_bench/reference_baselines.py

python vlm_bench/astro_modalities.py             # render image/text/physio
python vlm_bench/astro_eval.py encode --model clip
python vlm_bench/astro_eval.py encode --model biomedclip
python vlm_bench/astro_eval.py table1
python vlm_bench/astro_eval.py table2_astro
python vlm_bench/astro_eval.py table3
python vlm_bench/astro_eval.py table4
```

Encoding is the only step needing a GPU or a model download; everything after
it reads cached `.npz` vectors and runs on CPU in seconds.

### The large models

CLIP and BiomedCLIP encode on a laptop. BLIP-2, LLaVA-1.6, Qwen2.5-VL and
Med-Flamingo were encoded on an AMD MI300X (`ssh amdcluster`, torch
2.9.1+rocm6.3, bf16):

```bash
rsync -az --exclude data/raw --exclude .git ./ amdcluster:/root/astro-health/
ssh amdcluster 'cd /root/astro-health &&
  python3 -m venv --system-site-packages venv_astro &&
  ./venv_astro/bin/pip install --no-deps torchvision==0.24.1+rocm6.3 \
      --index-url https://download.pytorch.org/whl/rocm6.3 &&
  ./venv_astro/bin/pip install einops open_flamingo &&
  ./venv_astro/bin/python vlm_bench/encode_cluster.py --model blip2'
rsync -az amdcluster:/root/astro-health/data/vlm_astro/*/encodings/ ...
```

Five things that cost time and are not obvious:

- The box's Python is **PEP 668 externally-managed**, and it is shared with
  other projects. A venv with `--system-site-packages` keeps the working ROCm
  torch and isolates the additions; `--break-system-packages` would not.
- It runs **at ~99% disk**. Run one model at a time and purge between them
  (`--purge-after`); two concurrent downloads filled the disk and produced a
  corrupt LLaVA cache plus a malloc abort that looked like a model bug.
  Reclaiming `pip cache purge` + `uv cache clean` freed ~44GB without touching
  anything installed.
- **Qwen2.5-VL's processor needs torchvision**, which was absent — and it must
  match the ROCm torch build (`torchvision==0.24.1+rocm6.3`), not PyPI's CUDA
  wheel.
- **BLIP-2 files its hidden states differently** — under
  `.language_model_outputs`, not `.hidden_states`.
- **Med-Flamingo's `Flamingo.forward()` takes no `output_hidden_states`
  kwarg**; it passes a fixed argument list to the language encoder. Set
  `model.lang_encoder.config.output_hidden_states` instead. `encode_cluster.py`
  handles both this and the BLIP-2 case.

`notebooks/astro_vlm_kaggle.ipynb` does the same job on Kaggle if the cluster
is unavailable.

## What these numbers do and do not support

They support: a reference benchmark on three real astronaut/analog corpora with
a non-circular label, a floor, a control and a ceiling for every row, complete
coverage of six models, and four clean negative results.

They do not support: any claim about clinical deterioration detection. The
label is flight phase, not diagnosis. Cohorts are 4–10 people. A single
held-out subject moves balanced accuracy by several points, and no confidence
intervals are reported — with k=4 subjects on Inspiration4, differences smaller
than roughly 0.1 balanced accuracy should not be interpreted. The RQ2 reversal
above is a concrete demonstration of that fragility: swapping which model
represents "general-purpose" flipped the conclusion.

**The result the study set out to find is not here.** H1 predicted that
healthcare-oriented VLMs would outperform general-purpose ones on astronaut
health tasks. On this evidence they do not, zero-shot prompting does not work
at all, and the pixels add nothing over the numbers they were rendered from.
The defensible paper is a benchmark-and-negative-result paper: here is the
first reference benchmark on open astronaut health data, here are the floors
and ceilings, and here is why the obvious multimodal approach fails on it.

The single highest-value next experiment is not another VLM. It is to test
whether the heatmap rendering is the bottleneck — feed the same panels to a
model that reads tables natively (a text-serialised LLM prompt, or a tabular
foundation model such as TabPFN) under the identical leave-one-subject-out
protocol. Table 4 already suggests where that lands: text beat image by a wide
margin (MCC 0.409 vs 0.190) on exactly the same information.
