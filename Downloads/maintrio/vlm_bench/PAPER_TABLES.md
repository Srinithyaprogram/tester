# Fill-in values for the paper's Table 1–4 schemas

Generated from `vlm_bench/results/*.csv`. Full analysis and caveats:
`vlm_bench/ASTRO_RESULTS.md`.

---

## ⚠️ Correct the Table 2 currently in the draft before anything else

The draft's Table 2 reads:

| Model | Dataset | Accuracy | … |
|---|---|---|---|
| BiomedCLIP | NASA OSDR (Inspiration4) | 0.572 | … |
| Med-Flamingo | NASA OSDR (Inspiration4) | 0.562 | … |
| CheXagent | NASA GeneLab | 0.654 | … |
| BioViL-T | NASA GeneLab | 0.680 | … |

**Those numbers were not measured on NASA data.** They come from
`vlm_bench/results/table2.csv`, which is chest-radiograph benchmarking:

| Row | Dataset actually used |
|---|---|
| BiomedCLIP 0.572 | ChestMNIST (MedMNIST v2, from NIH ChestX-ray14) |
| Med-Flamingo 0.562 | VQA-RAD (radiology visual question answering) |
| CheXagent 0.654 | NIH ChestX-ray14 |
| BioViL-T 0.680 | NIH ChestX-ray14 |

`vlm_bench/RESULTS.md` states this explicitly — that table was run as a
methodology comparison "explicitly decoupled from astronaut data", because at
the time no astronaut corpus here had usable inputs plus a defensible label.
Publishing those numbers under a NASA dataset label misattributes them.

Either relabel that table as the chest-X-ray benchmark it is, or replace it
with the astronaut-data Table 2 below. Both are real; only the pairing of
numbers to dataset names is wrong.

---

## A schema change worth making: add a Protocol column

Each model yields two very different numbers, and collapsing to one row hides
the paper's cleanest finding:

- **zero-shot** — the model's own prompt-based answer. This is what RQ1 asks
  literally, and it is where the models fail: most become constant classifiers.
- **linear probe (leave-one-subject-out)** — logistic regression on the frozen
  embedding. Asks whether the representation *contains* the signal.

Reporting only zero-shot makes every model look broken. Reporting only the
probe hides that prompting does not work at all. Report both.

---

## Table 1 (RQ1) — general-purpose VLMs

### Zero-shot

| Model | Dataset | Accuracy | Precision | Recall | F1-score | AUROC | MCC |
|---|---|---|---|---|---|---|---|
| CLIP | NASA OSDR (Inspiration4) | 0.643 | 0.636 | 0.875 | 0.737 | 0.677 | 0.251 |
| BLIP-2 | NASA OSDR (Inspiration4) | 0.571 | 0.571 | 1.000 | 0.727 | 0.349 | 0.000 |
| LLaVA-1.6 | NASA GeneLab (OSD-174) | 0.667 | 0.667 | 1.000 | 0.800 | 0.791 | 0.000 |
| Qwen2.5-VL | NASA GeneLab (OSD-174) | 0.333 | 0.000 | 0.000 | 0.000 | 0.664 | 0.000 |

### Linear probe, leave-one-subject-out

| Model | Dataset | Accuracy | Precision | Recall | F1-score | AUROC | MCC |
|---|---|---|---|---|---|---|---|
| CLIP | NASA OSDR (Inspiration4) | 0.786 | 0.812 | 0.812 | 0.812 | 0.844 | 0.562 |
| BLIP-2 | NASA OSDR (Inspiration4) | 0.750 | 0.800 | 0.750 | 0.774 | 0.776 | 0.496 |
| LLaVA-1.6 | NASA GeneLab (OSD-174) | 0.767 | 0.795 | 0.875 | 0.833 | 0.836 | 0.453 |
| Qwen2.5-VL | NASA GeneLab (OSD-174) | 0.767 | 0.825 | 0.825 | 0.825 | 0.811 | 0.475 |

An MCC of exactly 0.000 with recall 1.000 (or 0.000) is a constant classifier —
every sample assigned to one class. Three of the four zero-shot rows are that.
Report balanced accuracy or MCC in the text; plain accuracy makes LLaVA-1.6's
degenerate row read as 0.667.

---

## Table 2 (RQ2) — biomedical VLMs, on astronaut data

### Zero-shot

| Model | Dataset | Accuracy | Precision | Recall | F1-score | AUROC | MCC |
|---|---|---|---|---|---|---|---|
| BiomedCLIP | NASA OSDR (Inspiration4) | 0.429 | 0.000 | 0.000 | 0.000 | 0.396 | 0.000 |
| Med-Flamingo | NASA OSDR (Inspiration4) | 0.679 | 0.640 | 1.000 | 0.780 | 0.674 | 0.400 |
| CheXagent | NASA GeneLab (OSD-174) | 0.500 | 0.656 | 0.525 | 0.583 | 0.460 | −0.024 |
| BioViL-T | NASA GeneLab (OSD-174) | 0.333 | 0.500 | 0.025 | 0.048 | 0.559 | −0.066 |

### Linear probe, leave-one-subject-out

| Model | Dataset | Accuracy | Precision | Recall | F1-score | AUROC | MCC |
|---|---|---|---|---|---|---|---|
| BiomedCLIP | NASA OSDR (Inspiration4) | 0.607 | 0.647 | 0.688 | 0.667 | 0.651 | 0.190 |
| Med-Flamingo | NASA OSDR (Inspiration4) | 0.714 | 0.700 | 0.875 | 0.778 | 0.693 | 0.411 |
| CheXagent | NASA GeneLab (OSD-174) | 0.817 | 0.854 | 0.875 | 0.864 | 0.839 | 0.583 |
| BioViL-T | NASA GeneLab (OSD-174) | 0.633 | 0.661 | 0.925 | 0.771 | 0.573 | −0.047 |

**RQ2's hypothesis is not supported.** Comparing like with like (same corpus,
same protocol), best general-purpose vs best biomedical, paired subject
bootstrap: Inspiration4 +0.156 balanced accuracy favouring general-purpose
(p=0.009); GeneLab 0.000 (p=1.000); bed rest −0.014 (p=0.940). One significant
result, in the opposite direction to H1; two nulls.

Note also that the spread *within* the biomedical family (CheXagent 0.817 vs
BioViL-T 0.633 accuracy on GeneLab) exceeds the gap between families.

---

## Table 3 (RQ3) — cross-dataset generalisation

| Model | Training Dataset | Testing Dataset | Accuracy | Precision | Recall | F1-score | AUROC | MCC |
|---|---|---|---|---|---|---|---|---|
| CLIP | NASA OSDR (Inspiration4) | NASA GeneLab (OSD-174) | 0.650 | 0.694 | 0.850 | 0.764 | 0.568 | 0.122 |
| BLIP-2 | NASA GeneLab (OSD-174) | NASA OSDR (Inspiration4) | 0.536 | 0.560 | 0.875 | 0.683 | 0.609 | −0.067 |
| BiomedCLIP | NASA OSDR (Inspiration4) | NASA OSDR bed rest (OSD-370) | 0.633 | 0.623 | 0.943 | 0.750 | 0.530 | 0.219 |
| Med-Flamingo | NASA GeneLab (OSD-174) | NASA OSDR bed rest (OSD-370) | 0.483 | 0.643 | 0.257 | 0.367 | 0.562 | 0.067 |

**The schema says "NASA ALSDA"; these rows use OSD-370 instead.** No ALSDA
export exists — ALSDA's non-omics holdings surface through LSDA, which is
managed-access with no public API. The file in this repo that carried the name
`alsda_physiology` was fabricated by `scripts/make_example_panel.py` and is now
quarantined. OSD-370 (10 bed-rest subjects, vastus lateralis transcriptomics) is
the real openly-downloadable analog cohort standing in for it. The table must
say so.

Protocol: fit on every sample of the training corpus, predict every sample of
the test corpus. No subject-wise split is needed — the cohorts are disjoint by
construction, which is what makes this a generalisation test.

All four rows are at or below chance on balanced accuracy (0.479–0.571) against
within-corpus probes of 0.750–0.844. Transfer does not happen.

---

## Table 4 (RQ4) — ablation study

BiomedCLIP, NASA OSDR (Inspiration4), linear probe leave-one-subject-out.

| Model | Configuration | Dataset | Accuracy | Precision | Recall | F1-score | AUROC | MCC |
|---|---|---|---|---|---|---|---|---|
| BiomedCLIP | Image Only | NASA OSDR (Inspiration4) | 0.607 | 0.647 | 0.688 | 0.667 | 0.651 | 0.190 |
| BiomedCLIP | Text Only | NASA OSDR (Inspiration4) | 0.714 | 0.722 | 0.812 | 0.765 | 0.802 | 0.409 |
| BiomedCLIP | Physiological Data Only | NASA OSDR (Inspiration4) | 0.786 | 0.778 | 0.875 | 0.824 | 0.776 | 0.559 |
| BiomedCLIP | Image + Text + Physiological Data | NASA OSDR (Inspiration4) | 0.750 | 0.765 | 0.812 | 0.788 | 0.812 | 0.486 |

Step 1 of the schema says "select the best-performing biomedical VLM from
Table 2". On astronaut data that is now **CheXagent**, not BiomedCLIP. These
rows keep BiomedCLIP because it is the only biomedical model here with a
genuine separate text tower, which the Text Only configuration requires — a
generative VLM's text representation is not separable from its image
conditioning. State that choice rather than implying BiomedCLIP won Table 2.

Physiological > Text > Image, and fusion does not beat physiology alone on any
metric except AUROC.

---

## Numbers the paper needs but the schemas have no column for

Without these the tables are uninterpretable — a reader cannot tell 0.65 from a
strong result or from noise.

**Reference points** (leave-one-subject-out, balanced accuracy):

| | Inspiration4 (k=4) | GeneLab (k=10) | Bed rest (k=10) |
|---|---|---|---|
| Majority-class floor | 0.500 | 0.500 | 0.500 |
| Deviation-magnitude control | 0.646 | 0.725 | 0.517 |
| Raw-measurement ceiling | 0.667 † | 0.738 | 0.794 |

† 0.771 with all 605 features, but ~0.10 of that is a missing-data artifact —
one crew member is missing an entire 406-analyte immune panel at one preflight
draw. 0.667 is the complete-case figure. Use it.

**Confidence intervals** are wide enough to change the reading — best model per
corpus: Inspiration4 0.844 [0.719, 0.958]; GeneLab 0.800 [0.688, 0.900]; bed
rest 0.606 [0.463, 0.749].

**Bed rest is a null result.** Seven of eight MCC intervals contain zero; the
eighth (CLIP) excludes zero on the negative side. Nothing there beats chance.

**No ranking is stable.** Changing which models represent a family, or
switching to the stricter `label_acute`, flips the winner on two of three
corpora. Report the qualitative findings, not a leaderboard.

---

## What the label means, for the methods section

The label is flight phase taken verbatim from NASA's deposited
`Factor Value[Spaceflight]` / `Factor Value[Bed Rest]` — an **exposure proxy,
not a clinical deterioration diagnosis**. It is mission metadata and therefore
independent of the measurements, so it is not circular. But no result here
supports a claim about detecting clinical deterioration, and the paper should
not phrase it as one.
