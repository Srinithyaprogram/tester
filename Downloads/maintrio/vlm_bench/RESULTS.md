# Table 2 — Biomedical Vision-Language Models (RQ2), as actually run

RQ2 as originally scoped ("do biomedical VLMs beat general-purpose VLMs at
astronaut health monitoring?") could not be answered on the datasets named in
the proposal: NASA OSDR (Inspiration4) and NASA GeneLab contain zero imaging
data (audited live — see the main repo's `docs/OSDR_FINDINGS.md`), and none of
it carries a clinical outcome label. All four models here are Vision-Language
Models that require images; none exists in the astronaut corpora.

**Decision made:** benchmark each model on the labeled public imaging dataset
it was actually built/evaluated on, and report RQ2 as a methodology
comparison, explicitly decoupled from astronaut data. That's what's below.

## Final Table 2

| Model | Dataset | Accuracy | Precision | Recall | F1 | AUROC | MCC |
|---|---|---|---|---|---|---|---|
| BiomedCLIP | ChestMNIST (derived from NIH ChestX-ray14) | 0.572 | 0.605 | 0.416 | 0.493 | 0.624 | 0.152 |
| Med-Flamingo | VQA-RAD | 0.562 | 0.567 | 0.288 | 0.382 | 0.590 | 0.108 |
| CheXagent | NIH ChestX-ray14 | 0.654 | 0.608 | 0.868 | 0.715 | 0.741 | 0.341 |
| BioViL-T | NIH ChestX-ray14 | 0.680 | 0.708 | 0.612 | 0.657 | 0.713 | 0.363 |

Every row is real inference on real, labeled, public data — not simulated,
not fabricated. n=500 per row (subsampled, seeded, stratified by label — see
below), except Med-Flamingo/VQA-RAD which used its full 251-example curated
closed-QA set (see `curate_datasets.py`; VQA-RAD's closed subset is smaller
than 500 to begin with).

**Reading this table:** on this evidence, the two CXR-specialist models
(BioViL-T, CheXagent) outperform the two more general biomedical models
(BiomedCLIP, Med-Flamingo) on binary abnormality detection — consistent with
domain-specialized pretraining helping on an in-domain task. CheXagent's high
recall / lower precision suggests it's biased toward predicting "finding
present." None of these numbers say anything about astronaut health
monitoring specifically — see the framing note above.

## Task framing

Every row is reduced to **binary classification** so all six requested
metrics are well-defined:

| Model | Positive-class signal |
|---|---|
| BiomedCLIP | zero-shot: softmax over cosine similarity to `"a chest x-ray with an abnormal finding"` vs `"...no abnormal findings"` |
| BioViL-T | same zero-shot contrastive scoring, via Microsoft's `hi-ml-multimodal` package (the HF `AutoModel` repo is text-only — see Model notes) |
| CheXagent | generative VQA, but scored via next-token logit P(yes)/(P(yes)+P(no)) rather than parsing free-form generation — keeps it a real probability for AUROC |
| Med-Flamingo | same yes/no-logit trick, zero-shot single-image prompting |

## Datasets

| Dataset | Used by | Real source | Notes |
|---|---|---|---|
| ChestMNIST | BiomedCLIP | MedMNIST v2 (derived from NIH ChestX-ray14) | Downloaded at native 28px, upsampled to 224px locally — the 224px source was a 3.9GB slow download that wasn't worth the wait for BiomedCLIP's own internal resizing |
| NIH ChestX-ray14 | CheXagent, BioViL-T | `alkzar90/NIH-Chest-X-ray-dataset` on HF, official metadata CSV + one raw image zip chunk | `datasets.load_dataset` doesn't work on this repo (legacy loading script, no Parquet conversion) — bypassed via direct `huggingface_hub.hf_hub_download` + manual zip parsing |
| VQA-RAD | Med-Flamingo | `flaviagiammarino/vqa-rad` | Restricted to closed yes/no questions (251/451) — open-ended questions have no fixed answer set, so Accuracy/F1/AUROC/MCC aren't defined for them |

All curation is in `curate_datasets.py`, all reproducible: rerun with
`python curate_datasets.py --dataset <name>`.

**Subsampling:** every dataset above 500 rows is subsampled to 500,
stratified by label, seed=2026 (`eval_models.py::_load_manifest`) — unbatched
single-image forward passes through multi-GB models were slow enough
(~9% GPU utilization observed, CPU-bound on preprocessing) that the full sets
(22,433 / 3,362 images) would have taken hours. Explicit, logged at runtime,
not a silent truncation.

## Model notes (things that weren't obvious going in)

- **BioViL-T's HF repo is text-only.** `AutoModel.from_pretrained("microsoft/BiomedVLP-BioViL-T")`
  loads `CXRBertModel` — confirmed by introspection, only exposes
  `get_projected_text_embeddings`, no image methods at all. The real
  image+text zero-shot pipeline is Microsoft's separate `hi-ml-multimodal`
  package (`health_multimodal.vlp.ImageTextInferenceEngine`).
- **Med-Flamingo's declared base model is gone.** `decapoda-research/llama-7b-hf`
  (named in the med-flamingo repo) 404s — taken down. Used `huggyllama/llama-7b`,
  the standard community replacement (same LLaMA-1 7B weights).
- **Med-Flamingo here is zero-shot, not the original 6-shot demo prompt** —
  replicating that exactly needs specific few-shot demo images not available
  here. Documented deviation, not a silent one.
- **CheXagent's remote-code model file has a long tail of undeclared
  dependencies** not in its model card: `transformers==4.40.0` exactly (hard
  assertion, conflicts with the `transformers` version everything else in
  this bench uses), plus `cv2`, `torchvision`, `einops`, `albumentations`,
  `sentencepiece`, `protobuf` — none listed anywhere, all surfaced one at a
  time as each import failed. Isolated into its own venv
  (`venv_chexagent`, `--system-site-packages` to keep the working ROCm torch)
  rather than risk breaking the other three models' shared environment.

## Reproducing

```bash
# datasets
python curate_datasets.py --dataset chestmnist
python curate_datasets.py --dataset nih_cxr14 --n-per-class 500
python curate_datasets.py --dataset vqarad
python curate_datasets.py --dataset pathvqa   # curated but not in the final table -- see below

# main venv: biomedclip, biovilt, medflamingo
python eval_models.py --model biomedclip
python eval_models.py --model biovilt
python eval_models.py --model medflamingo

# chexagent needs its own venv (see Model notes)
python -m venv --system-site-packages venv_chexagent
source venv_chexagent/bin/activate
pip install --no-deps 'transformers==4.40.0' 'tokenizers>=0.19,<0.20'
pip install --no-deps 'torchvision==0.24.1+rocm6.3' --index-url https://download.pytorch.org/whl/rocm6.3
pip install --no-deps einops
pip install albumentations opencv-python-headless sentencepiece protobuf
python eval_models.py --model chexagent
```

Path-VQA (3,362 curated image-QA pairs) was curated as Med-Flamingo's second
native benchmark but not run — VQA-RAD alone was sufficient for one table
row within available time; rerunning `eval_medflamingo("pathvqa")` is a
one-line change (`MODEL_EVALUATORS["medflamingo"]`'s dataset arg).
