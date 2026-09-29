#!/usr/bin/env python3
"""
Encode astronaut panels with a VLM, then produce Tables 1, 3 and 4.

Two stages, deliberately separated:

  encode   the only step that needs a GPU or a model download. Writes
           data/vlm_astro/<dataset>/encodings/<model>.npz -- a few hundred KB
           of per-sample vectors.
  table*   pure CPU. Reads the .npz files and computes every metric, transfer
           result and ablation.

That split means the expensive pass runs once (locally for the small models,
on Kaggle for the large ones -- drop the .npz files into the same directory)
and every downstream analysis is reproducible on a laptop in seconds.

    python vlm_bench/astro_modalities.py          # first: render the modalities
    python vlm_bench/astro_eval.py encode --model clip
    python vlm_bench/astro_eval.py encode --model biomedclip
    python vlm_bench/astro_eval.py table1
    python vlm_bench/astro_eval.py table3
    python vlm_bench/astro_eval.py table4

Two protocols are reported for every model, and they answer different questions:

  zero_shot      the model's own prompt-based prediction, no fitting at all.
                 This is what "how well does CLIP do on astronaut health"
                 literally means, and it is the honest headline number.
  linear_probe   logistic regression on the frozen embedding, leave-one-
                 subject-out. This asks whether the representation *contains*
                 the signal even when the prompt cannot elicit it -- a much
                 more informative result for a benchmark paper, and the only
                 protocol under which cross-dataset transfer (RQ3) is defined.

Compare both against vlm_bench/reference_baselines.py, which gives the
majority-class floor and the raw-feature ceiling for the same splits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ASTRO_ROOT = REPO_ROOT / "data" / "vlm_astro"
RESULTS = Path(__file__).resolve().parent / "results"

DATASETS = ("inspiration4", "genelab_astronaut", "osdr_bedrest")

#: Which models this script can encode itself. Everything else is expected to
#: arrive as a .npz produced by notebooks/astro_vlm_kaggle.ipynb.
LOCAL_MODELS = ("clip", "biomedclip")

#: Display names and the dataset each table row is specified against.
TABLE1_ROWS = (
    ("clip", "inspiration4"),
    ("blip2", "inspiration4"),
    ("llava16", "genelab_astronaut"),
    ("qwen25vl", "genelab_astronaut"),
)
TABLE3_ROWS = (
    ("clip", "inspiration4", "genelab_astronaut"),
    ("blip2", "genelab_astronaut", "inspiration4"),
    ("biomedclip", "inspiration4", "osdr_bedrest"),
    ("medflamingo", "genelab_astronaut", "osdr_bedrest"),
)
TABLE4_MODEL = "biomedclip"
TABLE4_DATASET = "inspiration4"

PRETTY = {
    "clip": "CLIP", "blip2": "BLIP-2", "llava16": "LLaVA-1.6",
    "qwen25vl": "Qwen2.5-VL", "biomedclip": "BiomedCLIP",
    "medflamingo": "Med-Flamingo", "biovilt": "BioViL-T", "chexagent": "CheXagent",
}
DATASET_PRETTY = {
    "inspiration4": "NASA OSDR (Inspiration4)",
    "genelab_astronaut": "NASA GeneLab (OSD-174)",
    "osdr_bedrest": "NASA OSDR bed rest (OSD-370)",
}


# --------------------------------------------------------------------------
# manifests and caches
# --------------------------------------------------------------------------

def load_manifest(dataset: str) -> dict:
    path = ASTRO_ROOT / dataset / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no manifest for '{dataset}' -- run: python vlm_bench/astro_modalities.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def encoding_path(dataset: str, model: str) -> Path:
    return ASTRO_ROOT / dataset / "encodings" / f"{model}.npz"


def load_encoding(dataset: str, model: str) -> dict[str, np.ndarray] | None:
    path = encoding_path(dataset, model)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray | None = None,
            threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
        precision_score, recall_score, roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_pred is None:
        y_pred = (y_score >= threshold).astype(int)

    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0,
    }
    out["auroc"] = (
        roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float("nan")
    )
    out.update(confusion_counts(y_true, y_pred))
    return {k: float(v) for k, v in out.items()}


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Confusion matrix as four named counts.

    Carried on every row because the scalar metrics hide the failure mode these
    models actually exhibit: a constant classifier reads as accuracy 0.667 and
    F1 0.800 on a 2:1 corpus, and only fp=0/fn=0 with one empty column makes it
    obvious that nothing was discriminated.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def _classifier(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Same regularisation as reference_baselines.py so the probe rows and the
    # raw-feature ceiling are comparable rather than differently tuned.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.01, max_iter=5000, random_state=seed),
    )


def linear_probe_loso(x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold scores from leave-one-subject-out logistic regression."""
    from sklearn.base import clone
    from sklearn.model_selection import LeaveOneGroupOut

    pred = np.zeros(len(y), dtype=int)
    score = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in LeaveOneGroupOut().split(x, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            only = int(y[train_idx][0])
            pred[test_idx], score[test_idx] = only, float(only)
            continue
        fitted = clone(_classifier(seed)).fit(x[train_idx], y[train_idx])
        pred[test_idx] = fitted.predict(x[test_idx])
        score[test_idx] = fitted.predict_proba(x[test_idx])[:, 1]
    return pred, score


# --------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------

def _images_and_text(manifest: dict):
    from PIL import Image

    samples = manifest["samples"]
    images = [Image.open(REPO_ROOT / s["image_path"]).convert("RGB") for s in samples]
    texts = [s["text"] for s in samples]
    return samples, images, texts


def _features(output):
    """Unwrap CLIP feature calls across transformers versions.

    transformers 4.x returns a bare tensor from get_text_features /
    get_image_features; 5.x returns a BaseModelOutputWithPooling whose
    pooler_output holds the projected features. Kaggle and this laptop are on
    different majors, so both shapes have to work.
    """
    return output if hasattr(output, "norm") else output.pooler_output


def _unit(tensor):
    return tensor / tensor.norm(dim=-1, keepdim=True)


def encode_clip(manifest: dict, device: str) -> dict[str, np.ndarray]:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(name)
    samples, images, texts = _images_and_text(manifest)

    with torch.inference_mode():
        prompt_inputs = processor(
            text=list(manifest["prompts"]), return_tensors="pt", padding=True
        ).to(device)
        prompt_embed = _unit(_features(model.get_text_features(**prompt_inputs)))

        image_embed, text_embed = [], []
        for start in range(0, len(images), 32):
            batch = processor(images=images[start:start + 32], return_tensors="pt").to(device)
            emb = _features(model.get_image_features(**batch))
            image_embed.append(_unit(emb).cpu().numpy())

            tok = processor(
                text=texts[start:start + 32], return_tensors="pt",
                padding=True, truncation=True, max_length=77,
            ).to(device)
            temb = _features(model.get_text_features(**tok))
            text_embed.append(_unit(temb).cpu().numpy())

    image_embed = np.concatenate(image_embed)
    text_embed = np.concatenate(text_embed)
    logits = image_embed @ prompt_embed.cpu().numpy().T * float(model.logit_scale.exp())
    zeroshot = _softmax_positive(logits)
    return _pack(samples, image_embed, text_embed, zeroshot)


def encode_biomedclip(manifest: dict, device: str) -> dict[str, np.ndarray]:
    import open_clip
    import torch

    tag = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, _, preprocess = open_clip.create_model_and_transforms(tag)
    tokenizer = open_clip.get_tokenizer(tag)
    model = model.to(device).eval()
    samples, images, texts = _images_and_text(manifest)

    with torch.inference_mode():
        prompt_embed = model.encode_text(tokenizer(list(manifest["prompts"])).to(device))
        prompt_embed = prompt_embed / prompt_embed.norm(dim=-1, keepdim=True)

        image_embed, text_embed = [], []
        for start in range(0, len(images), 32):
            batch = torch.stack(
                [preprocess(im) for im in images[start:start + 32]]
            ).to(device)
            emb = model.encode_image(batch)
            image_embed.append((emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy())

            temb = model.encode_text(tokenizer(texts[start:start + 32]).to(device))
            text_embed.append((temb / temb.norm(dim=-1, keepdim=True)).cpu().numpy())

    image_embed = np.concatenate(image_embed)
    text_embed = np.concatenate(text_embed)
    logits = image_embed @ prompt_embed.cpu().numpy().T * float(model.logit_scale.exp())
    zeroshot = _softmax_positive(logits)
    return _pack(samples, image_embed, text_embed, zeroshot)


def _softmax_positive(logits: np.ndarray) -> np.ndarray:
    """P(positive class) from a (n, 2) logit array whose column 1 is positive."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities[:, 1]


def _pack(samples, image_embed, text_embed, zeroshot) -> dict[str, np.ndarray]:
    return {
        "ids": np.array([s["id"] for s in samples]),
        "labels": np.array([s["label"] for s in samples], dtype=int),
        "subjects": np.array([s["subject_id"] for s in samples]),
        "image_embed": image_embed.astype(np.float32),
        "text_embed": text_embed.astype(np.float32),
        "physio": np.array([s["physio"] for s in samples], dtype=np.float32),
        "zeroshot_score": zeroshot.astype(np.float32),
    }


ENCODERS = {"clip": encode_clip, "biomedclip": encode_biomedclip}


def run_encode(model: str, datasets: list[str], device: str) -> int:
    if model not in ENCODERS:
        print(f"'{model}' is not locally encodable. Available: {sorted(ENCODERS)}")
        print("Large models are encoded by notebooks/astro_vlm_kaggle.ipynb;")
        print("copy its .npz output into data/vlm_astro/<dataset>/encodings/.")
        return 1

    for dataset in datasets:
        manifest = load_manifest(dataset)
        print(f"  {model} x {dataset} ({manifest['n_samples']} samples) ...", end="", flush=True)
        packed = ENCODERS[model](manifest, device)
        out = encoding_path(dataset, model)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **packed)
        auroc = metrics(packed["labels"], packed["zeroshot_score"])["auroc"]
        print(f" done (zero-shot AUROC {auroc:.3f}) -> {out.relative_to(REPO_ROOT)}")
    return 0


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

def _row(model: str, dataset: str, protocol: str, data: dict, scores: np.ndarray,
         preds: np.ndarray | None = None, **extra) -> dict:
    return {
        "model": PRETTY.get(model, model),
        "dataset": DATASET_PRETTY.get(dataset, dataset),
        "protocol": protocol,
        "n_samples": len(data["labels"]),
        "n_subjects": int(len(np.unique(data["subjects"]))),
        **extra,
        **metrics(data["labels"], scores, preds),
    }


def table1(seed: int) -> pd.DataFrame:
    rows = []
    for model, dataset in TABLE1_ROWS:
        data = load_encoding(dataset, model)
        if data is None:
            rows.append({
                "model": PRETTY.get(model, model),
                "dataset": DATASET_PRETTY.get(dataset, dataset),
                "protocol": "not_run",
                "status": "missing_encoding",
            })
            continue
        rows.append(_row(model, dataset, "zero_shot", data, data["zeroshot_score"],
                         status="completed"))
        preds, scores = linear_probe_loso(
            data["image_embed"], data["labels"], data["subjects"], seed
        )
        rows.append(_row(model, dataset, "linear_probe_loso", data, scores, preds,
                         status="completed"))
    return pd.DataFrame(rows)


def table2_astro(seed: int) -> pd.DataFrame:
    """RQ2 on astronaut data: does biomedical pretraining actually help here?

    vlm_bench/RESULTS.md answers RQ2 on public medical imaging benchmarks
    because the astronaut corpora carry no images. This table asks the question
    directly instead -- same corpora, same rendering, same protocol, one
    general-purpose encoder against one biomedical encoder. Every pair is
    matched, so the difference is attributable to pretraining rather than to
    the task.
    """
    families = (
        ("clip", "general-purpose"),
        ("blip2", "general-purpose"),
        ("llava16", "general-purpose"),
        ("qwen25vl", "general-purpose"),
        ("biomedclip", "biomedical"),
        ("medflamingo", "biomedical"),
        ("biovilt", "biomedical"),
        ("chexagent", "biomedical"),
    )
    rows = []
    for dataset in DATASETS:
        for model, family in families:
            data = load_encoding(dataset, model)
            if data is None:
                rows.append({
                    "model": PRETTY.get(model, model), "family": family,
                    "dataset": DATASET_PRETTY.get(dataset, dataset),
                    "protocol": "not_run", "status": "missing_encoding",
                })
                continue
            rows.append(_row(model, dataset, "zero_shot", data, data["zeroshot_score"],
                             family=family, status="completed"))
            preds, scores = linear_probe_loso(
                data["image_embed"], data["labels"], data["subjects"], seed
            )
            rows.append(_row(model, dataset, "linear_probe_loso", data, scores, preds,
                             family=family, status="completed"))
    return pd.DataFrame(rows)


def table3(seed: int) -> pd.DataFrame:
    """Fit on every sample of the training corpus, predict every sample of the test one.

    No subject-wise split is needed: the cohorts are disjoint by construction,
    which is exactly what makes this a generalisation test rather than a
    within-cohort one.
    """
    from sklearn.base import clone

    rows = []
    for model, train_key, test_key in TABLE3_ROWS:
        train = load_encoding(train_key, model)
        test = load_encoding(test_key, model)
        if train is None or test is None:
            rows.append({
                "model": PRETTY.get(model, model),
                "train_dataset": DATASET_PRETTY.get(train_key, train_key),
                "test_dataset": DATASET_PRETTY.get(test_key, test_key),
                "protocol": "not_run",
                "status": "missing_encoding",
            })
            continue

        fitted = clone(_classifier(seed)).fit(train["image_embed"], train["labels"])
        scores = fitted.predict_proba(test["image_embed"])[:, 1]
        preds = fitted.predict(test["image_embed"])
        rows.append({
            "model": PRETTY.get(model, model),
            "train_dataset": DATASET_PRETTY.get(train_key, train_key),
            "test_dataset": DATASET_PRETTY.get(test_key, test_key),
            "protocol": "linear_probe_transfer",
            "status": "completed",
            "n_train": len(train["labels"]),
            "n_test": len(test["labels"]),
            **metrics(test["labels"], scores, preds),
        })
    return pd.DataFrame(rows)


def table4(seed: int) -> pd.DataFrame:
    """Modality ablation on the best biomedical VLM, same probe throughout."""
    data = load_encoding(TABLE4_DATASET, TABLE4_MODEL)
    if data is None:
        return pd.DataFrame([{
            "model": PRETTY[TABLE4_MODEL],
            "dataset": DATASET_PRETTY[TABLE4_DATASET],
            "protocol": "not_run",
            "status": "missing_encoding",
        }])

    def unit(a: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.where(norm < 1e-8, 1.0, norm)

    image, text = data["image_embed"], data["text_embed"]
    physio = data["physio"]
    configurations = {
        "Image Only": image,
        "Text Only": text,
        "Physiological Data Only": physio,
        "Image + Text + Physiological Data": np.hstack(
            [unit(image), unit(text), unit(physio)]
        ),
    }

    rows = []
    for name, features in configurations.items():
        preds, scores = linear_probe_loso(features, data["labels"], data["subjects"], seed)
        rows.append({
            "model": PRETTY[TABLE4_MODEL],
            "configuration": name,
            "dataset": DATASET_PRETTY[TABLE4_DATASET],
            "protocol": "linear_probe_loso",
            "status": "completed",
            "n_features": int(features.shape[1]),
            "n_samples": len(data["labels"]),
            **metrics(data["labels"], scores, preds),
        })
    return pd.DataFrame(rows)


SHOW = ["accuracy", "precision", "recall", "f1", "auroc", "mcc",
        "tn", "fp", "fn", "tp"]


def _report(frame: pd.DataFrame, name: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.csv"
    frame.to_csv(out, index=False)
    lead = [c for c in frame.columns if c not in SHOW and c != "status"]
    with pd.option_context("display.width", 240, "display.max_columns", 60):
        print(frame[lead + [c for c in SHOW if c in frame.columns]].to_string(
            index=False, float_format=lambda v: f"{v:.3f}", na_rep="--"
        ))
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    encode = sub.add_parser("encode", help="run a model and cache its per-sample vectors")
    encode.add_argument("--model", required=True)
    encode.add_argument("--dataset", choices=DATASETS, help="default: all")
    encode.add_argument("--device", default=None, help="cuda / mps / cpu")

    for name in ("table1", "table2_astro", "table3", "table4"):
        table = sub.add_parser(name, help=f"build {name}")
        table.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args(argv)

    if args.command == "encode":
        device = args.device
        if device is None:
            import torch

            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        datasets = [args.dataset] if args.dataset else list(DATASETS)
        print(f"Encoding with {args.model} on {device}")
        return run_encode(args.model, datasets, device)

    builder = {"table1": table1, "table2_astro": table2_astro,
               "table3": table3, "table4": table4}[args.command]
    _report(builder(args.seed), args.command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
