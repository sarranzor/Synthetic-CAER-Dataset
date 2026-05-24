# SynthCAER Dataset Generation

Synthetic context-aware emotion recognition dataset created for the Master's in Data Science thesis at Universitat Oberta de Catalunya. SynthCAER is generated using **Stable Diffusion XL** and covers **7 emotion classes** across **529 scene contexts**, producing one photorealistic image per emotion–context combination.

## Overview

| Property | Value |
|---|---|
| **Emotion classes** | 7 (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) |
| **Scene contexts** | 529 unique categories |
| **Images per seed** | 3,703 (529 × 7) |
| **Resolution** | 1024 × 1024 px |
| **Generation model** | Stable Diffusion XL 1.0 |

## Repository structure

```
.
├── synthcaer_categories.ipynb  # Build the unified context list from SUN397, Places365, WIDER
├── synthcaer_generation.ipynb  # Generate images via SDXL + build the dataset index
├── contexts.json               # 529 unique scene contexts (output of categories notebook)
└── requirements.txt
```

> **Note:** generated images (`SynthCAER/`) avaiable in [ZENODO](https://doi.org/10.5281/zenodo.20306477)

## Pipeline

### Step 1 — Context categories (`synthcaer_categories.ipynb`)

Merges scene-category lists from three sources and deduplicates them:

| Source | Categories |
|---|---|
| SUN397 | 397 |
| Places365 | 365 |
| WIDER | 61 |
| **Unique (after dedup)** | **529** |

The merged list is saved as `contexts.json`.

### Step 2 — Image generation (`synthcaer_generation.ipynb`)

For each (context, emotion) pair, SDXL receives a **dual prompt**:

- `prompt` — emotion modifiers + quality tags + person anchor (`one person facing camera, face clearly visible, medium shot`)
- `prompt_2` — same emotion modifiers + context-aware scene description (`recognizable {context} background`)

A shared negative prompt suppresses common failure modes (blurry images, empty scenes, hidden faces, extreme viewpoints).

**Generation settings:**

| Parameter | Value |
|---|---|
| Scheduler | Euler Discrete |
| Inference steps | 30 |
| Guidance scale | 9.0 |
| Seed | 42 |
| Precision | FP16 (GPU) / FP32 (CPU) |

## Dataset layout

```
SynthCAER/
└── seed_42/
    ├── 0000_abbey/
    │   ├── Angry.png
    │   ├── Disgust.png
    │   ├── Fear.png
    │   ├── Happy.png
    │   ├── Sad.png
    │   ├── Surprise.png
    │   └── Neutral.png
    ├── 0001_aerobics/
    │   └── ...
    ├── ...
    └── generation_results.parquet   # Full index with prompts and file paths
```

## Setup

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended. SDXL requires approximately **10–12 GB VRAM** at FP16 with `enable_model_cpu_offload`; CPU generation is functional but very slow.

## Workflow

1. **Build context list** — run `synthcaer_categories.ipynb` to produce `contexts.json` (or use the file already included).
2. **Generate images** — run `synthcaer_generation.ipynb`. Set `MAX_CONTEXTS` to a smaller number for a quick test run. Set `SKIP_EXISTING = True` to resume an interrupted run without overwriting completed images.
3. The notebook saves a `generation_results.parquet` index inside the seed folder and exports a per-context image grid for visual inspection.
