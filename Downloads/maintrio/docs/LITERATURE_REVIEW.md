# Brainstorming & Lit Review

We will use this doc to put down resources we find and key points, ideas and thoughts towards our literature review until we can narrow it down for a finalized proposal by Week 4. When you add links, make sure to also include the title and at least a few words about the paper.

---

## Proposed Research Theme
**Benchmarking General-Purpose and Domain-Specialized Vision-Language Models Across Medical and Celestial Image Analysis Tasks**

### Motivation
Recent Vision-Language Models (VLMs) have demonstrated strong performance across a wide range of multimodal tasks. However, it remains unclear whether domain-specialized VLMs consistently outperform general-purpose VLMs within their target domains and how well such models generalize beyond their specialized training distributions.

This project aims to investigate the effectiveness, robustness, and generalization capabilities of general-purpose and domain-specialized VLMs using datasets from two distinct image domains:
1. Medical imaging
2. Celestial body / planetary imagery

The long-term goal is to understand whether domain specialization provides meaningful advantages and whether those advantages transfer across domains.

---

## Potential Research Questions
*   **RQ1:** Do domain-specialized Vision-Language Models outperform general-purpose VLMs on tasks within their target domain?
*   **RQ2:** How well do medical-domain VLMs generalize to celestial image analysis tasks?
*   **RQ3:** How well do celestial-domain VLMs generalize to medical image analysis tasks?
*   **RQ4:** Which VLM architectures demonstrate the strongest cross-domain robustness?
*   **RQ5:** What types of visual concepts are consistently challenging across both medical and celestial image domains?
*   **RQ6:** Does model specialization improve interpretability and confidence calibration?
*   **RQ7:** Can a unified benchmarking framework be developed for evaluating VLM performance across highly distinct scientific domains?

---

## Initial Benchmarking Framework

### General-Purpose VLM Candidates
*   **GPT-4o / GPT-4V** (Proprietary, state-of-the-art VLM)
*   **Qwen-VL** (Open-source series, strong multilingual and high-resolution support)
*   **LLaVA** (Open-source, standard baseline for instruction-tuned MLLMs)
*   **InternVL** (Open-source, scales vision foundation model to large parameter sizes)
*   **Gemini Vision / Gemini 1.5 Pro / Flash** (Strong multimodal reasoning)
*   **Molmo** (Recent high-efficiency open-source VLM series from AI2)

### Medical-Domain VLM Candidates
*   **Med-Flamingo** ([Paper Link](https://arxiv.org/abs/2307.15189)): *Med-Flamingo: a Multimodal Medical Few-shot Learner*. Based on OpenFlamingo-9B, adapted for few-shot medical VQA and rationale generation.
*   **LLaVA-Med** ([Paper Link](https://arxiv.org/abs/2306.00890)): *LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day*. Fine-tunes LLaVA on PubMed-sourced medical figure-caption pairs.
*   **RadFM** ([Paper Link](https://arxiv.org/abs/2308.02463)): *Towards Generalist Foundation Model for Radiology by Leveraging Web-scale 2D&3D Medical Data*. Specialized foundation model for 2D/3D radiology scans.
*   **Med-Gemma / MedGemma 1.5** ([Paper Link](https://arxiv.org/abs/2404.18416)): Google's specialized model series for medical text/image comprehension, clinical VQA, and high-dimensional imaging analysis.

### Celestial / Geological / Remote Sensing VLM Candidates
*   **GeoChat** ([Paper Link](https://arxiv.org/abs/2311.15826)): *GeoChat: A Grounded Large Vision-Language Model for Remote Sensing*. Fine-tuned from LLaVA-1.5, excels at object detection, referring expression comprehension, and image captioning on satellite/remotely-sensed images.
*   **EarthGPT** ([Paper Link](https://arxiv.org/abs/2401.16822)): *EarthGPT: A Universal Multi-modal Large Language Model for Remote Sensing Image Comprehension*. Merges optical, SAR, and infrared sensors with an instruction-following corpus.

---

## Candidate Datasets

### Medical Imaging
*   **MIMIC-CXR / ChestX-ray14:** Chest X-ray datasets featuring clinical reports or labels. Useful for classification and clinical report generation tasks.
*   **PadChest:** Large chest X-ray dataset containing diagnostic labels and annotations.
*   **MedMNIST:** A collection of 10 standardized biomedical image datasets for 2D and 3D classification, making it an easy-to-use benchmark.

### Celestial Body & Earth Remote Sensing Imaging
*   **NASA Mars Rover Imagery:** Multi-spectral raw imagery from Curiosity/Perseverance cameras (MASTCAM, Hazard Cameras, Navigation Cameras) documenting martian soil, rocks, and atmospheric anomalies.
*   **HiRISE / Moon Crater Detection Datasets:** Lunar Reconnaissance Orbiter (LRO) or Lunar Orbiter datasets containing labeled impact craters for detection and bounding box evaluation.
*   **PDS (Planetary Data System):** NASA's raw scientific data portal hosting images of Venus, Mars, outer planets, and asteroids.

---

## Literature Review Notes

### RISER
*   **Title:** *RISER: Retrieval-Based Self-Improvement for Large Language Models*
*   **Key Takeaways:**
    *   Benchmarking can identify performance gaps before proposing a new method.
    *   Method development should be justified by measurable deficiencies in baseline results.
    *   Strong evaluation design is critical for publication-quality research.
    *   Provides a useful example of transitioning from benchmarking to methodology.
*   **Potential Relevance:** Can guide experimental design and evaluation methodology.

### LLaVA-Med (biomedical VLM baseline)
*   **Title:** *LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day*
*   **Key Takeaways:**
    *   Adapts general-domain models using a curriculum learning pipeline: aligning biomedical vocabulary first (using figure-caption pairs), then mastering conversational semantics (using GPT-4-generated instruction-following data).
    *   Demonstrates that domain-specific alignment yields better performance on biomedical QA compared to general models.

### GeoChat (grounded remote sensing baseline)
*   **Title:** *GeoChat: A Grounded Large Vision-Language Model for Remote Sensing*
*   **Key Takeaways:**
    *   First VLM optimized specifically for remote-sensing imagery, pointing out the major "domain gap" when general VLMs are applied to satellite/top-down images.
    *   Introduces spatial coordinate grounding, enabling the VLM to return boundary coordinates for objects in the text.

---

## Literature Review Questions

### Benchmarking Perspective
1.  Are domain-specific VLMs consistently superior to general-purpose VLMs?
2.  Which evaluation metrics are most suitable across multiple domains?
3.  How should robustness and generalization be measured?

### Scientific Domain Perspective
1.  What visual reasoning challenges are common to both medical and planetary imagery? (e.g., fine-grained texture analysis, monochrome/false-color channels, low signal-to-noise ratios, scale variation).
2.  Can anomaly detection serve as a common task across domains? (e.g., detecting lesions in lungs vs. identifying craters or rover wheel anomalies on Mars).
3.  Can classification or detection tasks be unified into a single benchmark framework?

### Publication Perspective
1.  Is a pure benchmark paper sufficient for EACL 2027 Industry Track?
2.  What additional analysis would strengthen the contribution?
3.  Would a cross-domain robustness study provide a stronger industry narrative?

---

## Next Steps
1.  Identify candidate datasets.
2.  Identify candidate VLMs.
3.  Define a common prediction or detection task.
4.  Design an evaluation protocol.
5.  Run initial benchmark experiments.
6.  Analyze weaknesses of current models.
7.  Determine whether a lightweight method is necessary after benchmarking results are available.
