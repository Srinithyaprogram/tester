#!/usr/bin/env python3
"""
Generate a VLM benchmarking dataset containing Medical and Celestial image VQA pairs.

1. Downloads 10 chest X-ray images (5 normal, 5 pneumonia) using PneumoniaMNIST from MedMNIST.
2. Downloads 10 Mars landscape images (different cameras and rovers) using NASA's Mars Rover API.
3. Saves these files under data/benchmark/images/ and creates a unified data/benchmark/dataset.json.

Run it with:
    python scripts/generate_benchmark_dataset.py
"""

import json
import os
import sys
from pathlib import Path
import urllib.request
from PIL import Image

# Add repository root to python path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Target paths
BENCHMARK_DIR = REPO_ROOT / "data" / "benchmark"
IMAGES_DIR = BENCHMARK_DIR / "images"
MEDICAL_DIR = IMAGES_DIR / "medical"
CELESTIAL_DIR = IMAGES_DIR / "celestial"
DATASET_JSON = BENCHMARK_DIR / "dataset.json"


def download_medical_images(n_per_class=5) -> list[dict]:
    print("--- Fetching Medical Images from MedMNIST (PneumoniaMNIST) ---")
    try:
        from medmnist import PneumoniaMNIST
    except ImportError:
        print("Error: medmnist is not installed. Please pip install medmnist first.")
        return []

    # Prefer 224px CXRs — domain VLMs (CheXagent / BioViL-T) expect real-resolution inputs.
    # 28px MedMNIST thumbnails are too small for fair medical VLM evaluation.
    try:
        dataset = PneumoniaMNIST(split="train", download=True, size=224)
    except TypeError:
        dataset = PneumoniaMNIST(split="train", download=True, size=28)
    
    # Extract labels and indices
    # label 0 = normal, label 1 = pneumonia
    normal_indices = [i for i, label in enumerate(dataset.labels) if label[0] == 0]
    pneumonia_indices = [i for i, label in enumerate(dataset.labels) if label[0] == 1]
    
    selected_indices = normal_indices[:n_per_class] + pneumonia_indices[:n_per_class]
    
    MEDICAL_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    
    for i, idx in enumerate(selected_indices):
        img_array = dataset.imgs[idx]
        label = int(dataset.labels[idx][0])
        label_name = "normal" if label == 0 else "pneumonia"
        
        # Save image
        filename = f"pneumonia_{label_name}_{i}.png"
        filepath = MEDICAL_DIR / filename
        
        img = Image.fromarray(img_array)
        img.save(filepath)
        
        # Define questions/answers for VLM evaluation
        question = (
            "Based on this chest X-ray image, is there visual evidence of pneumonia? "
            "Respond with either 'Yes' or 'No' and explain your clinical reasoning."
        )
        ground_truth = "No" if label == 0 else "Yes"
        
        samples.append({
            "id": f"med_{i:03d}",
            "image_path": str(filepath.relative_to(REPO_ROOT)),
            "domain": "medical",
            "task_type": "classification",
            "question": question,
            "ground_truth": ground_truth,
            "metadata": {
                "dataset": "PneumoniaMNIST",
                "label_code": label,
                "label_name": label_name,
                "dimensions": f"{img.width}x{img.height}"
            }
        })
        print(f"Saved: {filepath.relative_to(REPO_ROOT)}")
        
    return samples


def download_celestial_images(n_photos=10) -> list[dict]:
    print("\n--- Fetching Celestial Images from NASA Image & Video Library API ---")
    
    # Query for "mars rover" images
    query = "mars%20rover"
    url = f"https://images-api.nasa.gov/search?q={query}&media_type=image"
    
    CELESTIAL_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ahm-research/1.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())
        
        items = data.get("collection", {}).get("items", [])
        if not items:
            print("Warning: No items returned by NASA Image API.")
            return []
            
        print(f"Total API items available: {len(items)}")
        
        count = 0
        for item in items:
            if count >= n_photos:
                break
                
            links = item.get("links", [])
            data_list = item.get("data", [])
            if not links or not data_list:
                continue
                
            metadata = data_list[0]
            title = metadata.get("title", "Mars Terrain")
            description = metadata.get("description", "Image of Mars surface from rover.")
            nasa_id = metadata.get("nasa_id", f"PIA_{count}")
            
            # Find the best image link
            photo_url = None
            for link in links:
                if link.get("render") == "image":
                    photo_url = link.get("href")
                    # Prefer smaller preview/thumb size for quick downloading
                    if "~small" in photo_url or "~thumb" in photo_url:
                        break
            
            if not photo_url:
                photo_url = links[0].get("href")
                
            if not photo_url or not photo_url.startswith("https://"):
                continue
                
            suffix = Path(photo_url).suffix or ".jpg"
            # Ensure filename is safe (remove spaces/special characters from nasa_id)
            safe_id = "".join([c for c in nasa_id if c.isalnum() or c in ("-", "_")])
            filename = f"nasa_celestial_{safe_id}{suffix}"
            filepath = CELESTIAL_DIR / filename
            
            # Download file
            print(f"Downloading photo {count+1}/{n_photos}: {title}...")
            img_req = urllib.request.Request(photo_url, headers={"User-Agent": "ahm-research/1.0"})
            with urllib.request.urlopen(img_req, timeout=30) as img_resp:
                img_data = img_resp.read()
            filepath.write_bytes(img_data)
            
            # Construct question/answer
            question = (
                f"Describe what is happening or what features are visible in this Mars planetary science image."
            )
            ground_truth = f"Title: {title}. Description: {description}"
            
            samples.append({
                "id": f"cel_{count:03d}",
                "image_path": str(filepath.relative_to(REPO_ROOT)),
                "domain": "celestial",
                "task_type": "qa",
                "question": question,
                "ground_truth": ground_truth,
                "metadata": {
                    "nasa_id": nasa_id,
                    "title": title,
                    "description": description,
                    "date_created": metadata.get("date_created"),
                    "source_url": photo_url
                }
            })
            count += 1
            print(f"Saved: {filepath.relative_to(REPO_ROOT)}")
            
    except Exception as exc:
        print(f"Failed to fetch Celestial images: {exc}")
        
    return samples


def main() -> int:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Get medical VQA samples
    medical_samples = download_medical_images(n_per_class=5)
    
    # 2. Get celestial VQA samples
    celestial_samples = download_celestial_images(n_photos=10)
    
    all_samples = medical_samples + celestial_samples
    
    if not all_samples:
        print("Error: No samples could be generated.")
        return 1
        
    # Write dataset.json
    DATASET_JSON.write_text(json.dumps(all_samples, indent=2), encoding="utf-8")
    print(f"\nSuccessfully generated unified dataset manifest at {DATASET_JSON.relative_to(REPO_ROOT)}")
    print(f"Total samples: {len(all_samples)} ({len(medical_samples)} medical, {len(celestial_samples)} celestial)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
