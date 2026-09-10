# Omni-DFINE

> **Omni-DFINE: Real-time Unified Vision-Language Prompting for Open-Vocabulary Detection**

---

## Table of Contents

1. [Motivation & Problem](#1-motivation--problem)
2. [Core Contributions](#2-core-contributions)
3. [Method Details](#4-method-details)
4. [Training Setup](#5-training-setup)
5. [Results](#6-results)
   - 6.1 [Zero-shot detection on COCO and LVIS](#61-zero-shot-detection-on-coco-and-lvis)
   - 6.2 [Efficiency / accuracy trade-off](#62-efficiency--accuracy-trade-off)
   - 6.3 [Deployment efficiency](#63-deployment-efficiency)
   - 6.4 [Cross-domain generalization on ODinW](#64-cross-domain-generalization-on-odinw)
   - 6.5 [Scaling and data ablation](#65-scaling-and-data-ablation)
6. [Citation](#14-citation)



## 1. Motivation & Problem

**Open-vocabulary object detection (OVOD)** aims to localize and recognize
arbitrary objects specified by *textual* or *visual* prompts beyond a
predefined category set. Unlike closed-set detection, OVOD requires the
model to align visual representations with open-domain semantic concepts,
posing two simultaneous challenges: cross-modal alignment **and**
computational efficiency.

Existing OVOD methods fall into two paradigms:

| Paradigm                                | Pros                                            | Cons                                                                    |
| --------------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------------- |
| YOLO-based / lightweight frameworks     | Real-time inference                             | Weak semantic robustness in complex open domains                        |
| DETR-based / foundation-model detectors | High accuracy via dense cross-modal interaction | Significant computational overhead, large pre-training data requirement |

## 2. Core Contributions

1. **A unified, scalable decoupled OVOD framework.** Omni-DFINE is a
   lightweight DETR-based detector that bridges the trade-off between
   real-time efficiency and open-vocabulary generalization through a
   systematic decoupling paradigm across **prompt encoding**, **fusion**,
   and **metric learning**. Three variants (Small, Medium, Large) support
   diverse deployment scenarios.

2. **Parameter-free structural visual prompting.** LSDPE leverages
   analytical spectral energy distributions to mitigate spatial
   quantization loss and capture sharp object boundaries — **without
   expensive spatial convolutions or learnable filters**.

3. **Interference-free modality routing.** LW-VTFE circumvents quadratic
   cross-attention cost by decomposing multi-modal features into orthogonal
   subspaces, effectively suppressing semantic over-smoothing and
   **reducing fusion parameters by over 70%**.

4. **Harmonic hyperspherical regularization.** The HHA loss establishes
   harmonically decaying margin geometries on the hypersphere that
   prevent representation collapse and inherently preserve hierarchical
   semantic taxonomies for zero-shot generalization.

---

## 3. System Architecture

```
        ┌──── Text prompt ──────┐
        │   "a cat", "a backpack"│
        └───────────┬───────────┘
                    │  BERT-base (text backbone, shared across S/M/L)
                    ▼
                Text features ─────┐
                                    │                ┌─────────┐
                                    │     ┌─────────▶│ LSDPE   │──┐
        ┌──── Image ────┐          │     │          │(ASD+CFM)│  │
        │  HGNetv2 P3.. │─Encoder─┼─────┘          └─────────┘  │
        │  P4, P5, P64  │          │                              │
        └──── Visual ───┘          │   ┌──── LW-VTFE ◀───────────┘
              prompt (RoI)─────────▶│   │  (MOD + CAG)
                                    ▼   ▼
                                  ┌─────────┐
                                  │  DFINE  │  ◀── 6-layer decoder, 500 queries
                                  │ Decoder │      (FDR box regression preserved)
                                  └────┬────┘
                                       │
                          ┌────────────┴────────────┐
                          │                         │
                  VFL + L1 + GIoU + FGL + DDF    HHA Loss
                  (D-FINE box losses)            (vision-text alignment
                                                 on unit hypersphere,
                                                 harmonic decaying margins)
```



## 3. Training Setup

### 4.1 Datasets

Omni-DFINE is trained exclusively on **publicly available** open-source
detection and grounding datasets (paper Table I):

| Dataset         | Type      | Texts | Images | Annotations |
| --------------- | --------- | -----:| ------:| -----------:|
| Objects365 (V1) | Detection | 365   | 609K   | 9,621K      |
| GQA             | Grounding | 387K  | 621K   | 3,681K      |
| Flickr30k       | Grounding | 94K   | 149K   | 641K        |

## 5. Results

### 5.1 Zero-shot detection on COCO and LVIS

**COCO val2017 and LVIS zero-shot benchmarks** (paper Table II):

| Variant        | Prompt Type | COCO AP | LVISminival APall | LVISminival APr |
| -------------- | ----------- | -------:| -----------------:| ---------------:|
| Omni-DFINE (S) | Text        | 41.8    | 28.3              | 23.3            |
| Omni-DFINE (M) | Text        | 43.9    | 30.8              | 26.0            |
| Omni-DFINE (L) | Text        | 46.1    | 33.1              | 28.5            |
| Omni-DFINE (L) | Text+Visual | 46.9    | 35.8              | 31.2            |

**Headline takeaways:**

* Omni-DFINE (L) with joint text-visual prompting achieves **46.9% AP on
  COCO**, outperforming the previous leading real-time model OV-DEIM-L by
  +1.0% COCO AP and +3.4% LVIS APall under comparable pre-training data.
* **Rare-category dominance on LVIS:** the HHA loss gives the largest
  gains on `APr` (rare categories), demonstrating that hierarchical
  harmonic margins preserve coarse-grained semantic shells and prevent
  representation collapse — exactly where zero-shot generalization matters
  most.
* Even compared with non-real-time heavyweight detectors (GLIP-T,
  Grounding DINO-T), Omni-DFINE narrows the accuracy gap significantly
  while maintaining real-time inference (under 50 ms).

### 5.2 Efficiency / accuracy trade-off

**Efficiency-accuracy Pareto frontier on COCO zero-shot** (paper Table III,
all results on a single NVIDIA A40 GPU with TensorRT FP16 inference):

| Method             | Params (M) ↓ | FLOPs (G) ↓ | FPS ↑    | Epochs | COCO AP ↑ |
| ------------------ | ------------:| -----------:| --------:| ------:| ---------:|
| **Non-Real-Time**  |              |             |          |        |           |
| GLIP-T (Swin-T)    | 232.0        | 499.0       | 6.0      | –      | 42.9      |
| GLIPv2-T (Swin-T)  | 232.0        | 499.0       | 6.0      | –      | 43.1      |
| Grounding DINO-T   | 172.0        | 342.0       | 9.4      | 30     | 48.4      |
| T-Rex2-T           | 175.0        | 351.0       | –        | 30     | 46.1      |
| **Real-Time**      |              |             |          |        |           |
| YOLO-Worldv2-S     | 13.3         | 39.1        | 47.4     | 200    | –         |
| YOLOE-V8-S         | 11.5         | 30.2        | 58.1     | 80     | 33.1      |
| YOLOE-11-S         | 10.8         | 28.5        | 60.3     | 80     | 34.2      |
| YOLOE-11-M         | 21.0         | 74.2        | 52.5     | 80     | 40.1      |
| YOLOE-11-L         | 46.0         | 160.8       | 44.2     | 80     | 42.8      |
| G-DINO 1.5 Edge    | 27.5         | 78.4        | –        | 17     | 42.6      |
| OV-DEIM-S          | 11.0         | 34.2        | 58.1     | 20     | 40.8      |
| OV-DEIM-M          | 20.0         | 76.1        | 50.1     | 20     | 43.3      |
| OV-DEIM-L          | 36.0         | 120.6       | 35.3     | 20     | 45.9      |
| **Omni-DFINE (S)** | **11.2**     | **32.5**    | **61.9** | **20** | **41.8**  |
| **Omni-DFINE (M)** | **23.5**     | **72.1**    | **54.9** | **20** | **43.9**  |
| **Omni-DFINE (L)** | **36.4**     | **118.3**   | **40.2** | **20** | **46.1**  |

**Key observations:**

* **Omni-DFINE (S) hits 61.9 FPS**, significantly outperforming
  YOLO-Worldv2-S (47.4 FPS) and YOLOE-11-S (60.3 FPS) — paired with a
  substantial accuracy leap of **+7.6% AP over YOLOE-11-S** (41.8% vs 34.2%).
* **Omni-DFINE (L) achieves 46.1% AP**, matching the heavy-weight
  T-Rex2-T (175.0M params) while using only **21% of its parameters** and
  **34% of its FLOPs** (118.3 G vs 351.0 G).
* **Omni-DFINE (M) surpasses OmDet-Turbo-T by +1.4% AP** while using only
  13% of its model size, demonstrating the exceptional efficiency of
  hierarchical harmonic alignment in distilling cross-modal knowledge into
  a compact architecture.

### 5.3 Deployment efficiency

GPU memory and latency on a single NVIDIA A40 GPU with FP16 precision
(paper Table IV):

| Model          | Batch=1 Mem (GB) | Batch=1 Lat (ms) | Batch=32 Mem (GB) | Batch=32 Lat (ms) |
| -------------- | ----------------:| ----------------:| -----------------:| -----------------:|
| Omni-DFINE (S) | 1.2              | 16.2             | 6.5               | 148.5             |
| Omni-DFINE (M) | 1.6              | 18.2             | 10.2              | 215.0             |
| Omni-DFINE (L) | 2.2              | 24.9             | 15.8              | 310.4             |

The very low memory footprint at batch=1 (1.2-2.2 GB) confirms
suitability for **edge deployment**, while batch=32 throughput is
suitable for high-throughput batch processing.

### 5.4 Cross-domain generalization on ODinW

**Zero-shot ODinW13 / ODinW35 mAP** (paper Table VII):

| Method             | Backbone  | Training Data      | ODinW13  | ODinW35  |
| ------------------ | --------- | ------------------ | --------:| --------:|
| GLIP-T             | Swin-T    | O365, GoldG, Cap4M | 46.5     | 19.6     |
| Grounding DINO-T   | Swin-T    | O365, GoldG, Cap4M | 49.8     | 22.3     |
| YOLO-Worldv2-L     | YOLOv8-L  | O365, GoldG, CC2M  | 38.2     | 16.4     |
| YOLOE-11-L         | YOLOv11-L | O365, GoldG        | 40.5     | 17.8     |
| G-DINO 1.5 Edge-L  | EffViT-L1 | O365, GoldG, V3Det | 45.8     | 19.6     |
| Dynamic-DINO       | EffViT-L1 | O365, GoldG, V3Det | 46.8     | 20.0     |
| **Omni-DFINE (L)** | HGNetv2   | **O365 only**      | **45.2** | 18.9     |
| **Omni-DFINE (L)** | HGNetv2   | O365, GoldG        | **47.1** | **20.2** |

### 6.5 Scaling and data ablation

| Model          | Prompt   | Params (M) | COCO AP | LVISmin APall | LVISmin APr | LVISmin APc | LVISmin APf |
| -------------- | -------- | ----------:| -------:| -------------:| -----------:| -----------:| -----------:|
| Omni-DFINE (S) | Text     | 11.2       | 40.1    | 28.3          | 23.3        | 27.1        | 31.5        |
| Omni-DFINE (M) | Text     | 23.5       | 42.1    | 30.8          | 26.0        | 29.6        | 33.9        |
| Omni-DFINE (L) | Text     | 36.4       | 44.5    | 33.1          | 28.5        | 31.8        | 36.2        |
| Omni-DFINE (S) | Visual-G | 11.2       | 38.8    | 26.9          | 21.9        | 25.9        | 29.9        |
| Omni-DFINE (M) | Visual-G | 23.5       | 40.9    | 29.2          | 24.3        | 28.2        | 32.2        |
| Omni-DFINE (L) | Visual-G | 36.4       | 42.8    | 31.2          | 26.3        | 30.1        | 34.0        |

---

## 7. Hyperparameter Sensitivity

The authors study sensitivity to the key hyperparameters using the S
variant (paper Table VI):

| Module | Hyperparameter | Value    | mAP      |
| ------ | -------------- | --------:| --------:|
| LSDPE  | `α` (energy)   | 0.85     | 41.5     |
| LSDPE  | `α`            | **0.90** | **41.8** |
| LSDPE  | `α`            | 0.95     | 41.6     |
| HHA    | `λ_ref`        | 0.2      | 41.6     |
| HHA    | `λ_ref`        | **0.3**  | **41.8** |
| HHA    | `λ_ref`        | 0.4      | 41.4     |

---

## 8. Related Work

* **RT-DETR** — first real-time end-to-end DETR for closed-set scenarios;
  improves speed via an efficient hybrid encoder.
* **RT-DETRv2** — optimizes the decoder and training strategy to remove
  deployment constraints.
* **RT-DETRv4** — enhances lightweight model accuracy with a knowledge
  distillation framework.
* **D-FINE** — redefines the bounding box regression task and improves
  localization accuracy through the FINE distribution refinement
  mechanism (the **base detector** Omni-DFINE builds upon).
* **DEIM** — accelerates convergence and boosts accuracy with dense
  matching and a dedicated loss.
* **DEIMv2** — optimizes the architecture on top of DEIM for a better
  performance-cost trade-off.

---

## 9. Citation

```bibtex
@inproceedings{dfine2024,
  title     = {D-FINE: Redefine Regression Task of DETRs as Fine-grained
               Distribution Refinement},
  author    = {Peng, Yansong and others},
  booktitle = {ICLR},
  year      = {2024},
}
```

---
