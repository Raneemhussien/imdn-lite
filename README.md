# IMDN-lite: Efficient Image Super-Resolution on CPU

**Course:** CISC 867 — Deep Learning  
**Team:** Farida Gaber · Rodina Khallaf · Raneem Alnaghy  
**Institution:** School of Computing, Queen's University, Kingston, Canada  
**Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/farida5gaber/IMDN-Lite)  

---

## Overview

IMDN-lite is a 97,060-parameter adaptation of the [Information Multi-Distillation Network (IMDN)](https://doi.org/10.1145/3343031.3351084) for ×2 single-image super-resolution (SISR), designed to run **entirely on CPU** without GPU acceleration.

Starting from the original IMDN, we reduce feature channels 64 → 32 and IMDB block count 6 → 3, yielding a model ~7× smaller that retains the Progressive Refinement Module (PRM) and Contrast-Aware Channel Attention (CCA). Trained on a 200-image DIV2K subset for 10,000 gradient steps, IMDN-lite achieves **34.08 dB PSNR / 0.9441 SSIM** on a 20-image validation set — surpassing bicubic by **+1.64 dB** and outperforming SRCNN trained under identical conditions by **+1.66 dB**.

---

## Architecture

![IMDN-lite Architecture](https://github.com/Raneemhussien/imdn-lite/blob/main/figures/IMDN-Architecture.jpeg)

IMDN-lite follows a `HEAD → BODY → TAIL` pipeline. Total trainable parameters: **97,060**.

```
LR Input (3×48×48)
    │
HEAD: Conv2d(3→32, 3×3)        ← global residual saved here
    │
BODY: 3× IMDB-lite Block
    │   ├─ PRM: 4× (Conv + LeakyReLU(0.05)), progressive channel splitting (24→16→8→0)
    │   │       concatenates four 8-channel portions → 32ch output
    │   └─ CCA: per-channel contrast attention via sum(std, mean) + 2× 1×1 Conv + Sigmoid
    │
IIC: concatenate HEAD + 3× IMDB outputs (128ch) → 1×1 Conv → 32ch
    │
lr_conv: Conv2d(32→32, 3×3)    + global residual from HEAD
    │
TAIL: Conv2d(32→12, 3×3) + PixelShuffle(×2) → clamp [0,1]
    │
SR Output (3×96×96)
```

| Property | Original IMDN | IMDN-lite (Ours) |
|---|---|---|
| Feature channels | 64 | 32 |
| IMDB blocks | 4 (paper) / 6 (code) | 3 |
| Parameters | ~694,000 | 97,060 (~7× fewer) |
| Training images | 800 DIV2K | 200 DIV2K |
| Training patches | ~3,000,000+ | 116,914 |
| Hardware | GPU | CPU only |
| Scale factors | ×2, ×3, ×4 | ×2 only |

---

## Results

### DIV2K Validation Set (20 images, best checkpoint step 9,700)

| Method | PSNR (dB) | SSIM | Params | Time (ms/img) |
|---|---|---|---|---|
| Bicubic | 32.44 | 0.9221 | — | 5.33 |
| SRCNN | 32.42 | 0.9287 | 69,251 | 668.10 |
| FSRCNN | 25.77 | 0.7764 | 24,683 | 264.93 |
| **IMDN-lite (Ours)** | **34.08** | **0.9441** | **97,060** | **723.87** |

### Performance Progression

| Training Step | PSNR (dB) | SSIM | Gain vs. Bicubic |
|---|---|---|---|
| 1,000 | 32.71 | 0.9302 | +0.27 dB |
| 3,000 | 33.43 | 0.9393 | +0.99 dB |
| 6,000 | 33.66 | 0.9409 | +1.22 dB |
| **10,000 (best: step 9,700)** | **34.08** | **0.9441** | **+1.64 dB** |

Neither PSNR nor SSIM plateaued at step 10,000 — the reported result is a **lower bound**.

### Generalization

| Evaluation Set | Bicubic | IMDN-lite | Gain |
|---|---|---|---|
| DIV2K validation (20 imgs) | 32.44 | 34.08 | +1.64 dB |
| DIV2K held-out test (20 imgs) | 31.65 | 33.18 | +1.53 dB |
| Set14 benchmark (14 imgs) | 28.31 | 29.83 | +1.52 dB |

IMDN-lite achieves positive gains over bicubic on **all** held-out images (20/20 DIV2K test, 14/14 Set14).

### Model Compression

| Method | PSNR (dB) | Model Size | ΔPSNR |
|---|---|---|---|
| FP32 (base) | 34.0778 | 394.8 KB | — |
| FP16 quantization | 34.0776 | 205.3 KB | −0.0002 dB |
| 15% magnitude pruning | 33.5883 | 394.8 KB | −0.49 dB |

### Comparison with Original IMDN

| Configuration | PSNR (dB) | Training Data | Steps | Hardware |
|---|---|---|---|---|
| Original IMDN | 38.00 (Set5) | 800 DIV2K | ~500,000 | GPU |
| **IMDN-lite (Ours)** | **34.08 (DIV2K val)** | **200 DIV2K** | **10,000** | **CPU** |

The −3.92 dB gap is fully attributable to the ~50× smaller training budget (~1% of original compute).

---

## Key Findings

The most significant finding is **methodological**: switching from epoch-based to steps-based LR scheduling produced a **+3.43 dB gain** with no change to architecture or data.

> The epoch-based 30-epoch run (1,310 min of CPU training on 116,914 patches) achieved only **+0.55 dB** over bicubic, versus **+1.64 dB** for the steps-based run.

Three factors determined performance:
1. **Training methodology** — steps-based scheduling was the single highest-impact change (+3.43 dB)
2. **Data volume** — a 6.4× patch increase (+2.73 dB) crossed the bicubic baseline for the first time
3. **Architecture efficiency** — PRM + CCA extract more useful features per gradient update than SRCNN/FSRCNN

---

## Training Run History

| Run | Patches | Schedule | F/B | Steps | Bicubic | PSNR | Gap |
|---|---|---|---|---|---|---|---|
| 1 | ~2,700† | Epoch | 32/3 | 30ep | 32.34 | 27.15 | −5.18 |
| 2 | ~4,500† | Epoch | 32/3 | 50ep | 32.34 | 25.16 | −7.17 |
| 3 | ~4,500† | Epoch | 64/4 | 50ep | 32.34 | 23.55 | −8.78 |
| 4 | ~4,500† | **Steps** | 32/3 | 500 | 32.34 | 28.59 | −3.74 |
| 5 | ~4,500† | Steps | 32/3 | 1,000 | 32.34 | 29.64 | −2.69 |
| 6 | 18,269 | Steps | 32/3 | 1,000 | 31.70 | 29.17 | −2.53 |
| **7** | **116,914** | **Steps** | **32/3** | **1,000** | **32.44** | **32.68** | **+0.24** |

† Stride = 96 misconfiguration (too few patches). Run 4 marks the epoch→steps transition: **+3.43 dB** over Run 3 under identical architecture and data.

---

## Ablation Studies

### Block Depth (32 channels, 100 steps, seed=42)

| Blocks | Params | PSNR (dB) | SSIM | Note |
|---|---|---|---|---|
| 1 | 42,132 | 26.48 | 0.7521 | Insufficient capacity |
| **3 (Ours)** | **97,060** | **27.15** | **0.7831** | Optimal |
| 5 | 151,988 | 26.68 | 0.7600 | Over-parameterized at 100 steps |

### Channel Width (3 blocks, 1,000 steps)

| Channels | Params | PSNR (dB) | SSIM | Note |
|---|---|---|---|---|
| 16 | 25,480 | 31.73 | 0.9168 | Underfitted |
| **32 (Ours)** | **97,060** | **32.42** | **0.9304** | Optimal for CPU |
| 48 | 214,752 | 33.18 | 0.9360 | +0.76 dB, 2.2× params |

---

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/Raneemhussien/imdn-lite.git
cd imdn-lite
pip install torch==2.2.0 scikit-image numpy matplotlib pillow
```

### 2. Data

Download preprocessed `.npy` patch files:

- [Patches — first 100 images (Source 1)](https://drive.google.com/drive/folders/1kZ8T9BqNz1-S7CbUZkeeUQvzxH2BtMO)
- [Patches — second 100 images (Source 2)](https://drive.google.com/drive/folders/1vLVZFC2aJC1D4jLZ4T-OdFOnRlHsMT47)

Or regenerate from the [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/):

```bash
python scripts/preprocess.py \
  --hr_dir data/DIV2K_HR/ \
  --lr_dir data/DIV2K_LR/ \
  --output_dir data/patches/ \
  --patch_size 96 \
  --stride 48 \
  --scale 2 \
  --seed 42
```

### 3. Train

```bash
python scripts/train.py \
  --patch_dir data/patches/ \
  --val_dir data/val/ \
  --steps 10000 \
  --batch_size 16 \
  --lr 2e-4 \
  --channels 32 \
  --blocks 3 \
  --seed 42 \
  --save_dir checkpoints/run_final/
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --val_dir data/val/ \
  --scale 2 \
  --border_shave 2
```

Expected output (best checkpoint, step 9,700): **PSNR = 34.08 dB, SSIM = 0.9441**

---

## Repository Structure

```
├── model/
│   └── imdn_lite.py                     # IMDN-lite architecture (PRM, CCA, IIC)
├── data/
│   └── splits.txt                       # All image IDs: train / val / held-out test
├── scripts/
│   ├── preprocess.py                    # Patch extraction and augmentation pipeline
│   ├── train.py                         # Steps-based training loop
│   └── evaluate.py                      # PSNR/SSIM evaluation with border shave
├── checkpoints/
│   └── best_model.pth                   # Best checkpoint (step 9,700, 97,060 params)
├── figures/
│   ├── architecture.png                 # IMDN-lite architecture pipeline diagram
│   └── project_timeline_dark.png        # Project execution timeline
├── results/
│   ├── CISC_867_Midterm_Report_Group1.pdf
│   ├── CISC_867_Group1_Report.pdf       # Final report
│   ├── results_summary.json             # Full metrics and training history
│   ├── config.json                      # Hyperparameter configuration
│   ├── training_curves.png
│   ├── visual_comparison.png
│   ├── inference_time.png
│   └── per_image_psnr.png
├── notebooks/
│   └── Efficient_Image_Super-Resolution_with_Lightweight_CNNs.ipynb
├── README.md
└── LOG.md
```

---

## Reproducibility

All randomness fixed with `seed=42` (Python, NumPy, PyTorch, PYTHONHASHSEED).

- **Framework:** PyTorch 2.2.0+cpu
- **Hardware:** Google Colab CPU + local CPU (VS Code)
- **Hyperparameters:** `results/config.json`
- **Full training history:** `results/results_summary.json`
- **Verification:** run `evaluate.py` with `checkpoints/best_model.pth` → PSNR = 34.08 dB, SSIM = 0.9441

> **Checkpoint note:** `best_model.pth` is the best-PSNR checkpoint (step 9,700). To resume training, use `latest_model.pth` (last completed step) — the best checkpoint may be from an earlier step than the most recently completed one.

---

## Project Timeline

![Project Timeline](https://github.com/Raneemhussien/imdn-lite/blob/main/figures/CISC_867_Group1_Timeline.png)

The project ran across four phases over six weeks (May–June 2026):

- **Phase 1 — Baseline Setup (May 4–15):** Raneem led IMDN paper review. Rodina handled DIV2K download, preprocessing, and assembling the 116,914-patch dataset. Farida implemented the bicubic baseline, IMDN-lite architecture, and training loop. All training runs 1–7 completed before the midterm submission (May 15).
- **Phase 2 — Analysis and Tuning (May 18–25):** Raneem led hyperparameter tuning and ablation design. Farida conducted cross-run comparability analysis. Rodina produced training figures and plots. Team observed Eid Al-Adha break (~May 25–June 1).
- **Phase 3 — Final Experiments (June 1–8):** Farida implemented SRCNN/FSRCNN baselines, extended training to 10,000 steps, FP16 quantization, and magnitude pruning. Rodina ran ablation studies, computed all metrics, and evaluated on DIV2K test set and Set14. Raneem led qualitative panel production and result interpretation.
- **Phase 4 — Delivery (June 4–11):** Farida produced the final notebook. Rodina managed the GitHub repository. Raneem led presentation preparation. All three contributed to final report writing and proofreading.

---

## Team Contributions

**Rodina Khallaf** — Data & Evaluation Lead: full preprocessing pipeline (modulo cropping, patch extraction, rotation augmentation, `.npy` serialization), stride misconfiguration identification and corrected reprocessing, final 116,914-patch dataset assembly, all ablation experiments, generalization evaluation on the held-out DIV2K test set and Set14, training figures and plots, GitHub repository management.

**Farida Gaber** — Model & Implementation Lead: complete IMDN-lite architecture (PRM, CCA, IIC, lr_conv), steps-based training loop, gradient clipping, reproducibility controls, two critical bug fixes (missing lr_conv layer; missing 2-pixel border shave in metric computation), all 7 training runs, SRCNN/FSRCNN baselines, extended training to 10,000 steps, FP16 quantization, magnitude pruning, CPU inference timing, final notebook.

**Raneem Alnaghy** — Analysis & Reporting Lead: hyperparameter tuning strategy, ablation study design, cross-run comparability analysis, midterm report, qualitative panel production, result interpretation, final report writing, proofreading, and presentation preparation.

---

## References

[1] Z. Hui, X. Gao, Y. Yang, and X. Wang, "Lightweight Image Super-Resolution with Information Multi-Distillation Network," ACM MM 2019. doi: 10.1145/3343031.3351084

[2] E. Agustsson and R. Timofte, "NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study," CVPR Workshops 2017. https://data.vision.ee.ethz.ch/cvl/DIV2K/

[3] C. Dong, C. C. Loy, K. He, and X. Tang, "Learning a Deep Convolutional Network for Image Super-Resolution," ECCV 2014.

[4] C. Dong, C. C. Loy, and X. Tang, "Accelerating the Super-Resolution Convolutional Neural Network," ECCV 2016.

[5] J. Liang et al., "SwinIR: Image Restoration Using Swin Transformer," ICCV Workshops 2021. arXiv:2108.10257

[6] Z. Zheng, "IMDN PyTorch Implementation (reference only)." https://github.com/Zheng222/IMDN

---

> DIV2K is released for non-commercial research use only. No commercial application of this dataset or derived models is intended.
