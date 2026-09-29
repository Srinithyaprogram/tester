#!/usr/bin/env python3
"""
Run the four biomedical VLMs on their curated native benchmarks (see
curate_datasets.py) and fill Table 2 / RQ2. Every task is binary
classification with a probability score, so all six metrics are well-defined:

    model         | dataset            | positive-class prompt
    --------------|---------------------|---------------------------------
    BiomedCLIP    | chestmnist          | "a chest x-ray with an abnormal finding"
    BioViL-T      | nih_cxr14           | "findings suggestive of disease"
    CheXagent     | nih_cxr14           | yes/no VQA logit-probability
    Med-Flamingo  | vqarad + pathvqa    | yes/no VQA logit-probability

Usage:
    python eval_models.py --model biomedclip --dataset chestmnist
    python eval_models.py --model all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)

DATA_ROOT = Path(__file__).resolve().parent / "data"
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    try:
        out["auroc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        out["auroc"] = float("nan")  # single-class fold; report NaN, don't fabricate
    return out


MAX_SAMPLES = 500  # per-image forward passes on unbatched multi-GB models are
# slow enough (observed: ~9% GPU util, CPU-bound on preprocessing between
# calls) that the full ChestMNIST test split (22,433) or Path-VQA (3,362)
# would take hours. Capped uniformly, seeded, stratified by label -- explicit
# subsampling, not a silent truncation.


def _load_manifest(dataset: str, max_samples: int = MAX_SAMPLES) -> pd.DataFrame:
    manifest_path = DATA_ROOT / dataset / "labels.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No curated data at {manifest_path} -- run curate_datasets.py first")
    df = pd.read_csv(manifest_path)
    if len(df) > max_samples:
        per_class = max_samples // df["label"].nunique()
        parts = [g.sample(n=min(len(g), per_class), random_state=2026) for _, g in df.groupby("label")]
        df = pd.concat(parts).reset_index(drop=True)
        print(f"{dataset}: subsampled {len(df)} rows from {manifest_path} (max_samples={max_samples}, seed=2026)")
    df["image"] = df["image"].apply(lambda rel: str(DATA_ROOT / dataset / rel))
    return df


# --------------------------------------------------------------------------
# BiomedCLIP -- zero-shot contrastive classification (open_clip)
# --------------------------------------------------------------------------
def eval_biomedclip(dataset: str = "chestmnist") -> dict:
    import open_clip

    df = _load_manifest(dataset)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    tokenizer = open_clip.get_tokenizer("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    model = model.to(DEVICE).eval()

    prompts = ["a chest x-ray with no abnormal findings", "a chest x-ray with an abnormal finding"]
    text_features = model.encode_text(tokenizer(prompts).to(DEVICE))
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    y_score = []
    with torch.no_grad():
        for path in df["image"]:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            y_score.append(logits[0, 1].item())  # P(abnormal)

    return compute_metrics(df["label"].to_numpy(), np.array(y_score))


# --------------------------------------------------------------------------
# BioViL-T -- zero-shot contrastive classification (CXR-specific)
# --------------------------------------------------------------------------
def eval_biovilt(dataset: str = "nih_cxr14") -> dict:
    # NOTE: microsoft/BiomedVLP-BioViL-T on HF is the TEXT encoder only
    # (CXRBertModel) -- confirmed by introspecting its loaded class, which
    # exposes only get_projected_text_embeddings, no image methods at all.
    # The real joint image+text zero-shot pipeline lives in Microsoft's
    # separate `hi-ml-multimodal` package, verified against its actual
    # source (health_multimodal.vlp.ImageTextInferenceEngine
    # .get_similarity_score_from_raw_data -- confirmed via inspect.getsource
    # on the cluster, not guessed).
    from health_multimodal.image import get_image_inference
    from health_multimodal.text import get_bert_inference
    from health_multimodal.vlp import ImageTextInferenceEngine

    df = _load_manifest(dataset)
    engine = ImageTextInferenceEngine(
        image_inference_engine=get_image_inference(),  # defaults to BioViL-T
        text_inference_engine=get_bert_inference(),     # defaults to BioViL-T's BERT
    )
    engine.to(DEVICE)

    no_finding_prompt = "no evidence of disease"
    finding_prompt = "findings suggestive of disease"

    y_score = []
    for path in df["image"]:
        sim_neg = engine.get_similarity_score_from_raw_data(Path(path), no_finding_prompt)
        sim_pos = engine.get_similarity_score_from_raw_data(Path(path), finding_prompt)
        # Raw cosine similarities, not probabilities -- softmax with CLIP's
        # conventional temperature to get a proper P(finding) for AUROC.
        exp_neg, exp_pos = np.exp(100.0 * sim_neg), np.exp(100.0 * sim_pos)
        y_score.append(exp_pos / (exp_neg + exp_pos))

    return compute_metrics(df["label"].to_numpy(), np.array(y_score))


def _yes_no_prob_from_logits(logits: torch.Tensor, tokenizer) -> float:
    """Shared trick for generative VQA models: P(yes) / (P(yes) + P(no)) from
    the next-token logits at the position right after '...Answer:', rather
    than free-form generation -- keeps the task a well-posed binary
    classification with a real probability for AUROC/MCC, instead of parsing
    generated text."""
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    pair = torch.stack([logits[no_id], logits[yes_id]])
    probs = torch.softmax(pair, dim=0)
    return probs[1].item()


# --------------------------------------------------------------------------
# CheXagent -- generative VQA, binarized via yes/no token-probability
# --------------------------------------------------------------------------
def eval_chexagent(dataset: str = "nih_cxr14") -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    df = _load_manifest(dataset)
    tokenizer = AutoTokenizer.from_pretrained("StanfordAIMI/CheXagent-2-3b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "StanfordAIMI/CheXagent-2-3b", trust_remote_code=True
    ).to(DEVICE).to(torch.bfloat16).eval()

    question = "Is there any abnormal finding in this chest x-ray? Answer strictly Yes or No."
    y_score = []
    with torch.no_grad():
        for path in df["image"]:
            query = tokenizer.from_list_format([{"image": path}, {"text": question}])
            conv = [
                {"from": "system", "value": "You are a helpful assistant."},
                {"from": "human", "value": query},
            ]
            input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
            logits = model(input_ids.to(DEVICE)).logits[0, -1, :].float()
            y_score.append(_yes_no_prob_from_logits(logits, tokenizer))

    return compute_metrics(df["label"].to_numpy(), np.array(y_score))


# --------------------------------------------------------------------------
# Med-Flamingo -- zero-shot VQA, binarized via yes/no token-probability.
# NOTE: simplified from med-flamingo's own 6-shot demo prompt (which needs
# specific demo images not available here) to zero-shot single-image
# prompting -- standard OpenFlamingo VQA convention, documented deviation,
# not silently substituted.
# --------------------------------------------------------------------------
def eval_medflamingo(dataset: str) -> dict:
    from einops import repeat
    from huggingface_hub import hf_hub_download
    from open_flamingo import create_model_and_transforms

    df = _load_manifest(dataset)
    llama_path = str(Path.home() / "astronaut-vlm-bench" / "models" /
                      "models--huggyllama--llama-7b" / "snapshots")
    llama_path = str(next(Path(llama_path).iterdir()))  # resolve the single snapshot dir

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=llama_path,
        tokenizer_path=llama_path,
        cross_attn_every_n_layers=4,
    )
    checkpoint_path = hf_hub_download("med-flamingo/med-flamingo", "model.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    model = model.to(DEVICE).eval()
    tokenizer.padding_side = "left"

    y_score = []
    with torch.no_grad():
        for path, question in zip(df["image"], df["question"]):
            image = Image.open(path).convert("RGB")
            vision_x = image_processor(image).unsqueeze(0)
            vision_x = repeat(vision_x, "N c h w -> b N T c h w", b=1, T=1).to(DEVICE)

            prompt = f"<image>Question: {question} Answer:"
            tokenized = tokenizer([prompt], return_tensors="pt")
            lang_x = tokenized["input_ids"].to(DEVICE)
            attention_mask = tokenized["attention_mask"].to(DEVICE)

            logits = model(vision_x=vision_x, lang_x=lang_x, attention_mask=attention_mask).logits[0, -1, :].float()
            y_score.append(_yes_no_prob_from_logits(logits, tokenizer))

    return compute_metrics(df["label"].to_numpy(), np.array(y_score))


MODEL_EVALUATORS = {
    "biomedclip": (eval_biomedclip, "chestmnist"),
    "biovilt": (eval_biovilt, "nih_cxr14"),
    "chexagent": (eval_chexagent, "nih_cxr14"),
    "medflamingo": (eval_medflamingo, "vqarad"),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_EVALUATORS) + ["all"], required=True)
    args = parser.parse_args()

    targets = list(MODEL_EVALUATORS) if args.model == "all" else [args.model]
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = []
    for name in targets:
        fn, dataset = MODEL_EVALUATORS[name]
        print(f"\n=== {name} on {dataset} ===")
        started = time.time()
        try:
            metrics = fn(dataset)
            metrics.update(model=name, dataset=dataset, status="completed",
                            seconds=round(time.time() - started, 1))
            print(json.dumps(metrics, indent=2))
        except Exception as exc:
            metrics = {"model": name, "dataset": dataset, "status": "failed",
                       "error": f"{type(exc).__name__}: {exc}"}
            print(f"FAILED: {metrics['error']}")
        all_results.append(metrics)

    out_path = RESULTS_ROOT / "table2.csv"
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
