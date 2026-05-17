# IMDN-lite: Efficient Image Super-Resolution on CPU

**Course:** CISC 867 — Deep Learning  
**Team:** Farida Gaber, Rodina Khalaf, Raneem Alnaghy  
**Institution:** School of Computing, Queen's University, Kingston, Canada  
**Repository:** https://github.com/Raneemhussien/imdn-lite  
**Midterm Report:** see `results/CISC_867_Midterm_Report_Group1.pdf`

---

## Problem

Single-image super-resolution (SISR) on CPU-only hardware. We implement and 
evaluate IMDN-lite, a 97,060-parameter adaptation of IMDN (Hui et al., 2019), 
trained on a 200-image DIV2K subset for ×2 upscaling without GPU acceleration.

---

## Method

IMDN-lite reduces the original IMDN from 64 to 32 feature channels and from 6 
to 3 IMDB blocks. Each IMDB block contains a Progressive Refinement Module (PRM) 
for staged feature distillation and a Contrast-Aware Channel Attention (CCA) 
module for per-channel recalibration. An IIC layer aggregates multi-scale 
features before sub-pixel upsampling via PixelShuffle(×2).

Architecture: `HEAD → 3× IMDB(PRM+CCA) → IIC → lr_conv → TAIL`  
Parameters: 97,060 (~7× fewer than original IMDN)

| Property | Original IMDN | IMDN-lite |
|---|---|---|
| Feature channels | 64 | 32 |
| IMDB blocks | 6 | 3 |
| Parameters | ~694,000 | 97,060 |
| Training images | 800 | 200 (116,914 patches) |
| Hardware | GPU | CPU only |

---

## Results

| Model | PSNR (dB) | SSIM | ms/img | Params |
|---|---|---|---|---|
| Bicubic | 32.44 | 0.9221 | 35.20 | — |
| IMDN-lite | 32.68 | 0.9298 | 4,150.86 | 97,060 |
| **Gain** | **+0.24** | **+0.0077** | — | — |

Evaluated on 20-image validation set, DIV2K ×2, Y-channel PSNR/SSIM,  
2-pixel border shave, Google Colab CPU, PyTorch 2.10.0+cpu.  
Note: result at 1,000 steps (~13.7% of one epoch) — lower bound, not converged.

---

## Data

- **Source:** DIV2K training set (https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- **Subset:** 200 images (seed=42 sampling), split into 180 training / 20 validation / 5 held-out test
- **Image IDs:** see `data/splits.txt`
- **Patches:** 116,914 training patch pairs (96×96 HR / 48×48 LR), stride=48, 4× rotation augmentation
- **Preprocessed .npy patches (Source 1 — first 100 images):** https://drive.google.com/drive/folders/1kZ8T9BqNz1-S7CbUZkeeUQvzxH2BtMO
- **Preprocessed .npy patches (Source 2 — second 100 images):** https://drive.google.com/drive/folders/1vLVZFC2aJC1D4jLZ4T-OdFOnRlHsMT47

To regenerate patches from scratch:
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

---

## How to Run

### 1. Install Dependencies
```bash
pip install torch==2.10.0 scikit-image numpy matplotlib pillow
```

### 2. Train
```bash
python scripts/train.py \
  --patch_dir data/patches/ \
  --val_dir data/val/ \
  --steps 1000 \
  --batch_size 16 \
  --lr 2e-4 \
  --channels 32 \
  --blocks 3 \
  --seed 42 \
  --save_dir checkpoints/run7/
```

### 3. Evaluate
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --val_dir data/val/ \
  --scale 2 \
  --border_shave 2
```

Expected output: PSNR = 32.68 dB, SSIM = 0.9298 on the 20-image validation set.

---

## Repository Structure
├── model/
│   └── imdn_lite.py                    # IMDN-lite architecture (PRM, CCA, IIC)
├── data/
│   └── splits.txt                      # All image IDs: train / val / held-out test
├── scripts/
│   ├── preprocess.py                   # Patch extraction and augmentation pipeline
│   ├── train.py                        # Steps-based training loop
│   └── evaluate.py                     # PSNR/SSIM evaluation with border shave
├── checkpoints/
│   └── best_model.pth                  # Best checkpoint (step 1000, 97,060 params)
├── results/
│   ├── CISC_867_Midterm_Report_Group1.pdf
│   ├── results_summary.json            # Full metrics and training history
│   ├── config.json                     # Hyperparameter configuration
│   ├── training_curves.png             # PSNR/SSIM/Loss vs steps
│   ├── visual_comparison.png           # Bicubic vs IMDN-lite vs HR crops
│   ├── inference_time.png              # CPU inference time analysis
│   └── per_image_psnr.png             # Per-image PSNR breakdown
├── notebooks/
│   └── Efficient_Image_Super-Resolution_with_Lightweight_CNNs.ipynb
├── README.md
└── LOG.md

---

## Reproducibility

- All randomness fixed with seed=42 (Python, NumPy, PyTorch, PYTHONHASHSEED)
- Hardware: Google Colab CPU runtime
- Framework: PyTorch 2.10.0+cpu
- Hyperparameters: see `results/config.json`
- Full training history: see `results/results_summary.json`
- To verify: run `evaluate.py` with `checkpoints/best_model.pth`  
  → should produce PSNR=32.68 dB, SSIM=0.9298 on the 20-image validation set

---

## References

[1] Z. Hui, X. Gao, Y. Yang, and X. Wang, "Lightweight Image Super-Resolution 
with Information Multi-Distillation Network," ACM MM 2019.  
doi: 10.1145/3343031.3351084

[2] E. Agustsson and R. Timofte, "NTIRE 2017 Challenge on Single Image 
Super-Resolution: Dataset and Study," CVPR Workshops 2017.  
https://data.vision.ee.ethz.ch/cvl/DIV2K/

[3] Z. Zheng, "IMDN PyTorch Implementation (reference only)."  
https://github.com/Zheng222/IMDN
