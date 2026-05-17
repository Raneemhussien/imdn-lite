# Development Log — IMDN-lite

**Project:** Efficient Image Super-Resolution with Lightweight CNNs  
**Team:** Farida Gaber, Rodina Khalaf, Raneem Alnaghy  
**Repository:** https://github.com/Raneemhussien/imdn-lite

---

## Week 1
**Dates:** 1 May 2025 – 7 May 2025  
**Goals:** Read IMDN paper; subset and preprocess DIV2K; implement bicubic baseline.

### Progress
- Read Hui et al. (2019) IMDN paper; identified PRM, CCA, IIC as key components
- Sampled 100 images from DIV2K using seed=42; partitioned into 90 train / 10 val
- Implemented preprocessing pipeline: modulo crop, stride-based patch extraction, 4-variant rotation augmentation
- Implemented bicubic baseline with Y-channel PSNR/SSIM and 2-pixel border shave
- Normalized all patches to [0,1] float32; saved as .npy files for training efficiency

### Key Decisions
- Chose stride=48 (50% overlap) for patch extraction to maximize content diversity
- Used L1 loss over L2 to avoid over-smoothed outputs, consistent with IMDN paper
- Fixed seed=42 across Python, NumPy, and PyTorch for full reproducibility
- Stored validation images as full images (no patching) so PSNR/SSIM are comparable to published results

### Issues Encountered
- Source 1 preprocessing script used stride=96 instead of stride=48, producing non-overlapping patches (18,269 instead of expected ~98,000). Identified post-training. Corrected in Source 2. Run 8 (planned) will evaluate the fully corrected uniform-stride dataset.
- Border shave was initially missing from PSNR computation — corrected before Run 4.

---

## Week 2
**Dates:** 8 May 2025 – 16 May 2025  
**Goals:** Implement IMDN-lite architecture; complete all training runs; midterm report.

### Progress
- Implemented full IMDN-lite architecture: PRM (4-stage channel splitting), CCA (std+mean contrast attention), IIC (multi-scale aggregation), lr_conv, PixelShuffle TAIL
- Verified parameter count: 97,060 (confirmed against saved checkpoint)
- Implemented steps-based training loop with gradient clipping (max norm=1.0) and checkpoint saving
- Expanded dataset to 200 images after Runs 1–5 failed to surpass bicubic baseline
- Completed all 7 training runs (see Table II in midterm report)
- Ran block-depth ablation (1, 3, 5 blocks) at 100 steps — 3 blocks confirmed optimal
- Measured CPU inference time: bicubic 35.20 ms/img, IMDN-lite 4,150.86 ms/img (117.9×)
- Submitted midterm report

### Key Decisions
- Switched from epoch-based to steps-based training after Run 2 showed LR decay collapse — produced +3.43 dB controlled gain in Run 4
- Kept 32 features / 3 blocks after Run 3 showed larger models underperform on limited data
- Used minimum inference time over 5 runs to reduce variance in timing measurements
- Used Gap column (model PSNR − bicubic PSNR) as normalized cross-run performance measure due to validation set inconsistency between Run 6 and Run 7

### Issues Encountered
- lr_conv layer missing from initial architecture; identified by comparing parameter count (87,812 vs expected 97,060). Fixed before Run 7.
- Run 6 bicubic baseline (31.70 dB) inconsistent with Run 7 (32.44 dB) on nominally same 20-image validation set. Root cause not isolated; Run 7 uses confirmed corrected loading pipeline.
- StepLR scheduler did not activate within 1,000-step budget (one epoch requires ~7,307 steps). LR remained constant at 2×10⁻⁴ throughout all runs. Steps-based decay planned for final report.

---

## Week 3
**Dates:** 17 May 2025 – 21 May 2025  
**Goals:** Extended training; channel-width ablation; LR sensitivity ablation; Run 8; GitHub setup.

### Progress
- [x] Set up GitHub repository and pushed initial files
- [x] Locked 5 held-out test image IDs: 29, 83, 314, 490, 757 — documented in data/splits.txt. These images are permanently withheld from all remaining hyperparameter decisions and checkpoint selection until final report evaluation only.
- [ ] Run extended training (10,000 steps), logging PSNR every 500 steps
- [ ] Run channel-width ablation: features ∈ {16, 32, 48} at 1,000 steps
- [ ] Run LR sensitivity ablation: lr ∈ {1×10⁻⁴, 2×10⁻⁴, 4×10⁻⁴} at 1,000 steps
- [ ] Run 8: evaluate on fully corrected uniform-stride dataset (stride=48 for both sources)

### Key Decisions
- To be documented as experiments are completed

### Issues Encountered
- To be documented as they arise

---

## Week 4
**Dates:** 22 May 2025 – 28 May 2025  
**Goals:** Final report; presentation; GitHub cleanup.

### Progress
- [ ] Evaluate best model on 5 held-out test images; report unbiased PSNR/SSIM
- [ ] Final report written and submitted
- [ ] Presentation prepared and rehearsed
- [ ] GitHub repository finalized with all results and reproducibility scripts

### Key Decisions
- To be documented

### Issues Encountered
- To be documented
