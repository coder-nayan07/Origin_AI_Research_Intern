# Prompted Segmentation for Drywall Quality Assurance

This repository contains a fine-tuned **CLIPSeg** model designed to segment structural anomalies (cracks) and drywall taping areas in construction images based on natural language prompts. This project was completed as part of a Drywall QA assignment.

**Author:** Nayan Kumar | 3rd Year Int. M.Tech, Mathematics and Computing | IIT (ISM) Dhanbad

---

## Performance Summary

The model was evaluated across a global dataset of **5,984** validation samples.

* **Mean IoU (mIoU):** 0.3874
* **Mean Dice Score:** 0.5297
* **Inference Speed:** ~50ms per image
* **Model Size:** ~603 MB

---

##  Visualizations

Below are representative qualitative results demonstrating the model's ability to switch contexts based on the prompt (e.g., `"segment crack"` vs. `"segment taping area"`). 

 **Note to user:** *Place your generated strip images (Orig | GT | Pred) in a `visualizations/` folder and update the image paths below.*

## 🖼️ Visualizations
The following examples demonstrate the model's performance across varied scenes. Each image below is a single horizontal strip containing the **Original Image**, the **Ground Truth Mask**, and the **Model Prediction**.

### 1. Crack Detection (`"segment crack"`)
| Scenario Example (Original | GT | Prediction) |
| :--- |
| ![Crack Example 1](visualizations/viz_crack_1.png) |
| ![Crack Example 2](visualizations/viz_crack_4.png) |

### 2. Drywall Taping Area (`"segment taping area"`)
| Scenario Example (Original | GT | Prediction) |
| :--- |
| ![Drywall Example 1](visualizations/viz_drywall_2.png) |
| ![Drywall Example 2](visualizations/viz_drywall_3.png) |

---

## 🛠️ Installation & Setup

1. **Environment:** Ensure you have Python 3.9+ and a CUDA-enabled GPU (tested on DGX-A100).
2. **Clone the repository:**
   ```bash
   git clone https://github.com/coder-nayan07/Origin_AI_Research_Intern
   ```
3. **Install Dependencies:**
   ```bash
   pip install torch torchvision transformers pillow opencv-python pycocotools tqdm
   ```

> **Reproducibility:** All training, evaluation, and data splitting were conducted using **Seed 42**.

---

## 📂 Data Preparation

The repository expects the following directory structure to properly handle the hybrid dataset and avoid ID collisions. We use a prefix strategy (`crack_` and `drywall_`) for isolation.

```text
├── cracks-1/                 # Dataset for "segment crack" prompt
├── Drywall-Join-Detect-2/    # Dataset for "segment taping area" prompt
├── data/
│   └── masks/                # Generated binary masks (e.g., crack_123__segment_crack.png)
├── visualizations/           # Folder to save orig | GT | pred strips
├── train.py                  # Training script
├── eval_complete.py          # Evaluation script
└── visualize.py              # Script to generate comparison images
```

---

## 🏃 How to Run

### 1. Training
To re-train the model with Dataset Isolation, run the training script. This fine-tunes the `rd64-refined` CLIPSeg model for 3 epochs.
```bash
python train.py
```

### 2. Global Evaluation
To compute the `mIoU` and `Dice` scores across the entire validation set (5,984 samples), run:
```bash
python eval_complete.py
```

### 3. Generate Visualizations
To generate the `Original | Ground Truth | Predicted` image strips for the report, run:
```bash
python visualize.py
```
*Outputs will be saved in the `visualizations/` directory.*

---

## 📝 Technical Notes & Failure Analysis

* **Hybrid Supervision:** The model seamlessly handles pixel-level polygon annotations for the cracks dataset and bounding-box level weak supervision for the drywall dataset.
* **Metric Discrepancies:** The taping area mIoU is mathematically lower (0.3418). This is largely due to **Labeling Mismatch**: the model accurately predicts the narrow drywall seams, but the ground truth annotations in the dataset are often broad bounding boxes.
* **Faint Structures:** Extremely low-contrast or hairline cracks are occasionally missed or only partially segmented depending on lighting variations.
```
