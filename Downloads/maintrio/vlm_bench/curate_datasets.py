#!/usr/bin/env python3
"""
Curate the native public benchmarks used for Table 2 / RQ2 (biomedical VLM
comparison). These replace the astronaut-health datasets originally named in
the table -- NASA OSDR/Inspiration4 and NASA GeneLab contain no imaging data
and no clinical outcome label (audited; see the main repo's
data/raw/EXAMPLE_DATA_README.md and docs/OSDR_FINDINGS.md) -- so the four
biomedical VLMs are instead benchmarked on the labeled imaging datasets they
were actually built for. RQ2 becomes a methodology comparison, decoupled from
astronaut data, and is reported as such.

Every task is framed as **binary classification** so all six requested
metrics (Accuracy/Precision/Recall/F1/AUROC/MCC) are well-defined:

    dataset        | models              | positive class
    ---------------|----------------------|----------------------------------
    ChestMNIST     | BiomedCLIP           | any of 14 chest findings present
    NIH ChestX-ray14 (subsample) | CheXagent, BioViL-T | any finding present
    VQA-RAD  (closed/yes-no subset) | Med-Flamingo | answer == "yes"
    Path-VQA (closed/yes-no subset) | Med-Flamingo | answer == "yes"

Usage:
    python curate_datasets.py --dataset chestmnist
    python curate_datasets.py --dataset nih_cxr14 --n-per-class 500
    python curate_datasets.py --dataset vqarad
    python curate_datasets.py --dataset pathvqa
    python curate_datasets.py --dataset all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATA_ROOT = Path(__file__).resolve().parent / "data"
SEED = 2026


def _write_manifest(dest_dir: Path, rows: list[dict], note: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(dest_dir / "labels.csv", index=False)
    print(f"wrote {dest_dir / 'labels.csv'}  ({len(manifest)} rows)")
    print(f"  positive rate: {manifest['label'].mean():.3f}")
    print(f"  note: {note}")


def curate_chestmnist() -> None:
    """BiomedCLIP zero-shot benchmark. Source: MedMNIST v2 (derived from NIH
    ChestX-ray14, resized to 224px here for BiomedCLIP's native input size).
    """
    import medmnist
    from medmnist import ChestMNIST

    dest_dir = DATA_ROOT / "chestmnist"
    images_dir = dest_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = DATA_ROOT / "_medmnist_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Default 28px source (~80MB) instead of the pre-resized 224px version
    # (~3.9GB) -- upsample ourselves below. Equivalent input to BiomedCLIP,
    # which resizes to 224 internally regardless of source resolution.
    test_set = ChestMNIST(split="test", download=True, size=28, root=str(cache_dir))
    info = medmnist.INFO["chestmnist"]
    label_names = list(info["label"].values())
    print(f"ChestMNIST test split: {len(test_set)} samples, {len(label_names)} finding labels")

    rows = []
    for i in range(len(test_set)):
        img, multi_label = test_set[i]
        # multi_label is a 14-way binary vector; "no finding" == all zero.
        any_finding = int(np.asarray(multi_label).sum() > 0)
        filename = f"chestmnist_{i:05d}.png"
        img.convert("RGB").resize((224, 224), Image.BICUBIC).save(images_dir / filename)
        rows.append({"image": f"images/{filename}", "label": any_finding})

    _write_manifest(
        dest_dir, rows,
        note="label=1 if any of the 14 CXR findings is present, 0 if 'no finding' "
             "(matches the binary framing used for NIH ChestX-ray14 below).",
    )


def _closed_yes_no_vqa(hf_name: str, dest_name: str) -> None:
    """Shared curation for Med-Flamingo's two native VQA benchmarks."""
    from datasets import load_dataset

    dest_dir = DATA_ROOT / dest_name
    images_dir = dest_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(hf_name, split="test")
    print(f"{hf_name} test split: {len(ds)} raw QA pairs")

    rows = []
    kept, skipped = 0, 0
    for i, example in enumerate(ds):
        answer = str(example["answer"]).strip().lower()
        if answer not in ("yes", "no"):
            skipped += 1
            continue
        filename = f"{dest_name}_{i:05d}.png"
        example["image"].convert("RGB").save(images_dir / filename)
        rows.append({
            "image": f"images/{filename}",
            "question": example["question"],
            "label": int(answer == "yes"),
        })
        kept += 1

    _write_manifest(
        dest_dir, rows,
        note=f"restricted to closed yes/no questions only ({kept} kept, {skipped} open-ended "
             f"questions dropped -- open-ended VQA has no fixed label set, so Accuracy/F1/AUROC/MCC "
             f"aren't well-defined for it). label=1 if ground-truth answer is 'yes'.",
    )


def curate_vqarad() -> None:
    """Med-Flamingo's own paper benchmark. Source: VQA-RAD (Lau et al. 2018),
    HF mirror flaviagiammarino/vqa-rad."""
    _closed_yes_no_vqa("flaviagiammarino/vqa-rad", "vqarad")


def curate_pathvqa() -> None:
    """Med-Flamingo's other native benchmark. Source: Path-VQA (He et al.
    2020), HF mirror flaviagiammarino/path-vqa."""
    _closed_yes_no_vqa("flaviagiammarino/path-vqa", "pathvqa")


def curate_nih_cxr14(n_per_class: int = 500) -> None:
    """CheXagent + BioViL-T benchmark (same dataset for both -- keeps the two
    CXR-foundation-model rows directly comparable). Source: NIH ChestX-ray14
    (Wang et al. 2017), HF mirror alkzar90/NIH-Chest-X-ray-dataset.

    Loaded via direct file download (huggingface_hub.hf_hub_download), NOT
    datasets.load_dataset -- this repo ships a legacy Python loading script,
    which datasets>=4.0 refuses to execute, and it has no HF auto-generated
    Parquet conversion (dataset viewer explicitly declines it, "runs
    arbitrary Python code"). The raw files are still directly fetchable:
    the official metadata CSV plus the images, chunked into 12 zips
    (~2-4GB each, ~42GB total). We pull only the metadata CSV + the smallest
    single zip chunk (images_001.zip, ~2GB, ~4-5k images) -- comfortably
    enough for an n_per_class-per-class subsample without the full corpus.

    Subsampled to n_per_class images per class (seeded), explicit not silent.
    Binary framing: 'No Finding' vs 'any of the 14 pathology labels present',
    same convention as ChestMNIST above.
    """
    import zipfile
    from huggingface_hub import hf_hub_download

    dest_dir = DATA_ROOT / "nih_cxr14"
    images_dir = dest_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    raw_cache = DATA_ROOT / "_nih_cxr14_raw"

    repo_id = "alkzar90/NIH-Chest-X-ray-dataset"
    csv_path = hf_hub_download(repo_id, "data/Data_Entry_2017_v2020.csv", repo_type="dataset",
                                local_dir=str(raw_cache))
    zip_path = hf_hub_download(repo_id, "data/images/images_001.zip", repo_type="dataset",
                                local_dir=str(raw_cache))

    metadata = pd.read_csv(csv_path)[["Image Index", "Finding Labels"]]
    label_of = dict(zip(metadata["Image Index"], metadata["Finding Labels"]))
    print(f"metadata: {len(metadata)} total NIH ChestX-ray14 image labels")

    rng = np.random.default_rng(SEED)
    target_each = n_per_class
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".png")]
        rng.shuffle(members)
        print(f"{zip_path}: {len(members)} images available")

        rows = []
        pos_count = neg_count = 0
        for member in members:
            fname = Path(member).name
            finding = label_of.get(fname)
            if finding is None:
                continue  # metadata CSV and this zip chunk mismatch -- skip, don't guess
            is_positive = finding != "No Finding"
            if is_positive and pos_count >= target_each:
                continue
            if not is_positive and neg_count >= target_each:
                continue
            with zf.open(member) as src:
                img = Image.open(src).convert("RGB")
            out_name = f"nih_{'pos' if is_positive else 'neg'}_{pos_count if is_positive else neg_count:05d}.png"
            img.save(images_dir / out_name)
            rows.append({"image": f"images/{out_name}", "label": int(is_positive)})
            if is_positive:
                pos_count += 1
            else:
                neg_count += 1
            if pos_count >= target_each and neg_count >= target_each:
                break

    print(f"collected {pos_count} 'any finding' + {neg_count} 'no finding' images "
          f"(target {target_each} each)")
    rng.shuffle(rows)
    _write_manifest(
        dest_dir, rows,
        note=f"subsampled to {target_each}/class (seed={SEED}) from the streamed test split, "
             f"not the full ~112k-image corpus. label=1 if any of 14 pathologies present, "
             f"0 if 'No Finding'.",
    )


CURATORS = {
    "chestmnist": curate_chestmnist,
    "vqarad": curate_vqarad,
    "pathvqa": curate_pathvqa,
    "nih_cxr14": curate_nih_cxr14,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(CURATORS) + ["all"], required=True)
    parser.add_argument("--n-per-class", type=int, default=500, help="nih_cxr14 only")
    args = parser.parse_args()

    targets = list(CURATORS) if args.dataset == "all" else [args.dataset]
    for name in targets:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        if name == "nih_cxr14":
            curate_nih_cxr14(n_per_class=args.n_per_class)
        else:
            CURATORS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
