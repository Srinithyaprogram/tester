#!/usr/bin/env python3
"""
Encode the large VLMs on a GPU box, one model at a time.

The counterpart to `astro_eval.py encode`, which only handles models small
enough for a laptop. Same output contract: one .npz per (model, dataset) under
data/vlm_astro/<dataset>/encodings/, so `astro_eval.py table1/table3` picks
them up with no further work.

    python vlm_bench/encode_cluster.py --model blip2
    python vlm_bench/encode_cluster.py --model llava16 --purge-after
    python vlm_bench/encode_cluster.py --all --purge-after

`--purge-after` deletes each model's HuggingFace cache entry once its
encodings are written. On a shared box sitting near 100% disk that is the
difference between finishing and an out-of-space failure halfway through.

Scoring follows vlm_bench/eval_models.py: generative VLMs are read through the
yes/no next-token logits rather than by parsing generated text, which keeps a
real probability for AUROC. The probe embedding is the mean-pooled final
hidden state.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
ASTRO_ROOT = REPO_ROOT / "data" / "vlm_astro"
DATASETS = ("inspiration4", "genelab_astronaut", "osdr_bedrest")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#: MI300X is happiest in bf16; it has no fp16 tensor-core advantage and bf16
#: avoids the overflow that fp16 hits on some of these vision towers.
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

MODEL_REPOS = {
    "blip2": ["Salesforce/blip2-opt-2.7b"],
    "llava16": ["llava-hf/llava-v1.6-mistral-7b-hf"],
    "qwen25vl": ["Qwen/Qwen2.5-VL-3B-Instruct"],
    "medflamingo": ["med-flamingo/med-flamingo", "huggyllama/llama-7b"],
    "biovilt": ["microsoft/BiomedVLP-BioViL-T"],
    "chexagent": ["StanfordAIMI/CheXagent-2-3b"],
}


def load_manifest(dataset: str) -> dict:
    path = ASTRO_ROOT / dataset / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"no manifest for '{dataset}' at {path}")
    manifest = json.loads(path.read_text())
    for sample in manifest["samples"]:
        sample["resolved"] = REPO_ROOT / sample["image_path"]
    return manifest


def question_for(manifest: dict) -> str:
    positive = manifest["prompts"][1]
    return (
        "This heatmap encodes one astronaut biomedical sample, one cell per "
        f"measurement. Is this {positive}? Answer Yes or No."
    )


def yes_no_probability(logits: torch.Tensor, tokenizer) -> float:
    """P(yes) / (P(yes) + P(no)) from the next-token logits."""
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    pair = torch.stack([logits[no_id], logits[yes_id]])
    return float(pair.float().softmax(dim=-1)[1])


def save(model: str, dataset: str, manifest: dict, scores, embeds,
         text_embeds=None) -> Path:
    from sklearn.metrics import roc_auc_score

    samples = manifest["samples"]
    out_dir = ASTRO_ROOT / dataset / "encodings"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model}.npz"
    embeds = np.asarray(embeds, dtype=np.float32)
    # A generative VLM has no separate text tower: the hidden state already
    # conditions on the prompt, so text_embed mirrors image_embed to keep the
    # file shape uniform. Dual encoders (BioViL-T) pass a real one.
    text = embeds if text_embeds is None else np.asarray(text_embeds, dtype=np.float32)
    np.savez_compressed(
        path,
        ids=np.array([s["id"] for s in samples]),
        labels=np.array([s["label"] for s in samples], dtype=int),
        subjects=np.array([s["subject_id"] for s in samples]),
        image_embed=embeds,
        text_embed=text,
        physio=np.array([s["physio"] for s in samples], dtype=np.float32),
        zeroshot_score=np.asarray(scores, dtype=np.float32),
    )
    labels = [s["label"] for s in samples]
    auroc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    print(f"    -> {path.relative_to(REPO_ROOT)}  (zero-shot AUROC {auroc:.3f})", flush=True)
    return path


def _hidden_states(out):
    """Final-layer hidden states, wherever this model files them.

    Most VLMs expose `.hidden_states` directly. BLIP-2 returns a
    Blip2ForConditionalGenerationModelOutput that keeps them one level down in
    `.language_model_outputs`, so a bare attribute access fails on it.
    """
    hidden = getattr(out, "hidden_states", None)
    if hidden is None:
        nested = getattr(out, "language_model_outputs", None)
        hidden = getattr(nested, "hidden_states", None)
    if hidden is None:
        raise AttributeError(
            f"{type(out).__name__} exposes no hidden_states; "
            f"available: {[f for f in getattr(out, 'keys', lambda: [])()]}"
        )
    return hidden


def _forward(model, inputs, tokenizer):
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True)
    score = yes_no_probability(out.logits[0, -1], tokenizer)
    embed = _hidden_states(out)[-1][0].float().mean(0).cpu().numpy()
    return score, embed


# --------------------------------------------------------------------------
# per-model encoders
# --------------------------------------------------------------------------

def encode_blip2(manifest: dict):
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    name = MODEL_REPOS["blip2"][0]
    processor = Blip2Processor.from_pretrained(name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        name, torch_dtype=DTYPE
    ).to(DEVICE).eval()

    question = question_for(manifest)
    scores, embeds = [], []
    for sample in manifest["samples"]:
        image = Image.open(sample["resolved"]).convert("RGB")
        inputs = processor(
            images=image, text=f"Question: {question} Answer:", return_tensors="pt"
        ).to(DEVICE, DTYPE)
        score, embed = _forward(model, inputs, processor.tokenizer)
        scores.append(score)
        embeds.append(embed)
    return model, scores, embeds


def encode_llava16(manifest: dict):
    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

    name = MODEL_REPOS["llava16"][0]
    processor = LlavaNextProcessor.from_pretrained(name)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        name, torch_dtype=DTYPE
    ).to(DEVICE).eval()

    question = question_for(manifest)
    scores, embeds = [], []
    for sample in manifest["samples"]:
        image = Image.open(sample["resolved"]).convert("RGB")
        inputs = processor(
            images=image, text=f"[INST] <image>\n{question} [/INST]", return_tensors="pt"
        ).to(DEVICE)
        score, embed = _forward(model, inputs, processor.tokenizer)
        scores.append(score)
        embeds.append(embed)
    return model, scores, embeds


def encode_qwen25vl(manifest: dict):
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenVL
    except ImportError:  # older transformers name
        from transformers import Qwen2VLForConditionalGeneration as QwenVL

    name = MODEL_REPOS["qwen25vl"][0]
    processor = AutoProcessor.from_pretrained(name)
    model = QwenVL.from_pretrained(name, torch_dtype=DTYPE).to(DEVICE).eval()

    question = question_for(manifest)
    scores, embeds = [], []
    for sample in manifest["samples"]:
        image = Image.open(sample["resolved"]).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(DEVICE)
        score, embed = _forward(model, inputs, processor.tokenizer)
        scores.append(score)
        embeds.append(embed)
    return model, scores, embeds


def encode_medflamingo(manifest: dict):
    """Med-Flamingo via open_flamingo.

    Its declared base model `decapoda-research/llama-7b-hf` was taken down;
    `huggyllama/llama-7b` is the standard community replacement carrying the
    same LLaMA-1 7B weights (documented in vlm_bench/RESULTS.md).

    Zero-shot single-image prompting, not the original 6-shot demo prompt --
    that needs specific demo images not available here.
    """
    from huggingface_hub import hf_hub_download
    from open_flamingo import create_model_and_transforms

    adapter, base = MODEL_REPOS["medflamingo"]
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=base,
        tokenizer_path=base,
        cross_attn_every_n_layers=4,
    )
    checkpoint = hf_hub_download(adapter, "model.pt")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
    model = model.to(DEVICE, DTYPE).eval()
    tokenizer.padding_side = "left"

    # Flamingo.forward() takes no output_hidden_states kwarg -- it passes a
    # fixed argument list through to the language encoder. Ask the encoder
    # itself instead, and keep a forward hook as a fallback for the case where
    # the wrapper drops the field on its way back out.
    model.lang_encoder.config.output_hidden_states = True
    captured: dict[str, torch.Tensor] = {}

    def grab(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach()

    handle = model.lang_encoder.get_input_embeddings().register_forward_hook(grab)

    question = question_for(manifest)
    scores, embeds = [], []
    try:
        for sample in manifest["samples"]:
            image = Image.open(sample["resolved"]).convert("RGB")
            vision_x = image_processor(image).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            vision_x = vision_x.to(DEVICE, DTYPE)
            lang_x = tokenizer(
                [f"<image>Question: {question} Answer:"], return_tensors="pt"
            ).to(DEVICE)
            with torch.inference_mode():
                out = model(
                    vision_x=vision_x,
                    lang_x=lang_x["input_ids"],
                    attention_mask=lang_x["attention_mask"],
                )
            scores.append(yes_no_probability(out.logits[0, -1], tokenizer))
            hidden = getattr(out, "hidden_states", None)
            vector = hidden[-1][0] if hidden else captured["hidden"][0]
            embeds.append(vector.float().mean(0).cpu().numpy())
    finally:
        handle.remove()
    return model, scores, embeds


def encode_biovilt(manifest: dict):
    """BioViL-T via Microsoft's hi-ml-multimodal package.

    Its HuggingFace repo is text-only -- `AutoModel.from_pretrained` loads a
    CXRBertModel with no image methods at all (see vlm_bench/RESULTS.md). The
    real image+text pipeline lives in `health_multimodal`, which downloads the
    image weights separately.

    Dual-encoder like BiomedCLIP, so scoring is prompt similarity rather than
    yes/no logits, and it gets a genuine separate text embedding.
    """
    from health_multimodal.image import get_image_inference
    from health_multimodal.image.utils import ImageModelType
    from health_multimodal.text import get_bert_inference
    from health_multimodal.text.utils import BertEncoderType

    image_engine = get_image_inference(ImageModelType.BIOVIL_T)
    text_engine = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)

    prompts = list(manifest["prompts"])
    prompt_embed = text_engine.get_embeddings_from_prompt(prompts, normalize=True)
    prompt_embed = np.asarray(prompt_embed.cpu() if hasattr(prompt_embed, "cpu") else prompt_embed)

    texts = [s["text"] for s in manifest["samples"]]
    text_embed = text_engine.get_embeddings_from_prompt(texts, normalize=True)
    text_embed = np.asarray(text_embed.cpu() if hasattr(text_embed, "cpu") else text_embed)

    image_embed = []
    for sample in manifest["samples"]:
        # get_projected_global_embedding wants a path, and its transform expects
        # a real file -- the manifest PNGs already are, so pass them straight in.
        embedding = image_engine.get_projected_global_embedding(Path(sample["resolved"]))
        vector = embedding.cpu().numpy() if hasattr(embedding, "cpu") else np.asarray(embedding)
        image_embed.append(vector / (np.linalg.norm(vector) + 1e-8))
    image_embed = np.stack(image_embed)

    logits = image_embed @ prompt_embed.T * 100.0
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return None, probabilities[:, 1], image_embed, text_embed


def encode_chexagent(manifest: dict):
    """CheXagent -- generative VQA, scored through the yes/no logits.

    Needs `transformers==4.40.0` exactly (its remote code asserts on the
    version) plus cv2 / albumentations / einops / sentencepiece / protobuf,
    none of which its model card lists. Run it from its own venv:

        python -m venv --system-site-packages venv_chexagent
        ./venv_chexagent/bin/pip install --no-deps 'transformers==4.40.0' \\
            'tokenizers>=0.19,<0.20'
        ./venv_chexagent/bin/pip install albumentations opencv-python-headless \\
            sentencepiece protobuf einops

    CheXagent-2 does **not** take images through a processor. `AutoProcessor`
    returns a bare `CheXagentTokenizer` with `image_processor = None`, so
    passing `images=[...]` is silently dropped with a "Keyword arguments not
    recognized" warning and the model scores the text alone -- every sample
    gets an identical score and AUROC comes out at exactly 0.500. That is what
    the first run of this produced; exactly 0.500 across three corpora is the
    tell.

    Images go in as `<|img|>path<|/img|>` tokens inside the prompt; the visual
    encoder opens the paths itself (`modeling_visual.py::encode`).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = MODEL_REPOS["chexagent"][0]
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    # float32, not bf16: its visual encoder loads and transforms images to
    # float32 internally and never casts them, so bf16 weights raise
    # "Input type (torch.cuda.FloatTensor) and weight type (CUDABFloat16Type)
    # should be the same". A 3B model in fp32 is nothing against 192GB of VRAM.
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(DEVICE).eval()

    question = question_for(manifest)
    scores, embeds = [], []
    for sample in manifest["samples"]:
        path = Path(sample["resolved"]).resolve()
        prompt = (
            f"USER: <|img|>{path}<|/img|>{question} ASSISTANT: <s>"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        score, embed = _forward(model, dict(inputs), tokenizer)
        scores.append(score)
        embeds.append(embed)

    if len(set(np.round(scores, 6))) == 1:
        raise RuntimeError(
            "CheXagent produced an identical score for every sample -- the "
            "images are not reaching the model. Check the <|img|> prompt "
            "format before trusting any number from this."
        )
    return model, scores, embeds


ENCODERS = {
    "blip2": encode_blip2,
    "llava16": encode_llava16,
    "qwen25vl": encode_qwen25vl,
    "medflamingo": encode_medflamingo,
    "biovilt": encode_biovilt,
    "chexagent": encode_chexagent,
}


def purge(model_key: str) -> None:
    """Drop this model's HF cache entries. Disk here is the binding constraint."""
    from huggingface_hub.constants import HF_HUB_CACHE

    for repo in MODEL_REPOS[model_key]:
        folder = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}"
        if folder.exists():
            size = sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
            shutil.rmtree(folder, ignore_errors=True)
            print(f"    purged {repo} ({size / 1e9:.1f} GB)", flush=True)


def run(model_key: str, datasets: list[str], purge_after: bool) -> None:
    for dataset in datasets:
        manifest = load_manifest(dataset)
        print(f"  {model_key} x {dataset} ({manifest['n_samples']} samples)", flush=True)
        produced = ENCODERS[model_key](manifest)
        # Dual encoders return a real text embedding as a fourth element.
        model, scores, embeds, *rest = produced
        save(model_key, dataset, manifest, scores, embeds, rest[0] if rest else None)
        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    if purge_after:
        purge(model_key)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(ENCODERS))
    parser.add_argument("--all", action="store_true", help="every model, in order")
    parser.add_argument("--dataset", choices=DATASETS, help="default: all")
    parser.add_argument(
        "--purge-after", action="store_true",
        help="delete each model's HF cache once its encodings are written",
    )
    args = parser.parse_args(argv)

    if not args.model and not args.all:
        parser.error("pass --model <name> or --all")

    models = sorted(ENCODERS) if args.all else [args.model]
    datasets = [args.dataset] if args.dataset else list(DATASETS)
    print(f"device={DEVICE} dtype={DTYPE}")
    if DEVICE == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    failed = []
    for model_key in models:
        try:
            run(model_key, datasets, args.purge_after)
        except Exception as exc:  # noqa: BLE001 -- one model must not sink the rest
            print(f"  {model_key}: FAILED -- {type(exc).__name__}: {exc}", flush=True)
            failed.append(model_key)

    if failed:
        print(f"\nfailed: {failed}. Their table rows stay missing_encoding.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
