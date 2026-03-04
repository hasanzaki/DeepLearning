# CLAUDE.md — DeepLearning Course (MCTA 4363)

This file governs all AI-assisted work on this repository. Read this before making any changes.

---

## Project Identity

- **Course**: MCTA 4363 Deep Learning
- **Institution**: Educational course repository
- **Primary audience**: Undergraduate students learning deep learning
- **Execution environment**: Google Colab (GPU, primarily Tesla T4/A100)
- **Framework**: PyTorch ≥ 2.6.0 + TorchVision ≥ 0.21.0

---

## Git Workflow

- **Active development branch**: `claude/analyze-github-repo-Pa8D0`
- **Always push to**: `claude/analyze-github-repo-Pa8D0` (direct push to `main` is protected)
- **Colab badge URLs**: All point to `claude/analyze-github-repo-Pa8D0` branch (updated in README.md)
- **Commit style**: Descriptive messages with phase prefix, e.g. `Phase 2: Add Grad-CAM to notebook 10`

---

## Notebook Conventions

### Cell Order (must follow this structure in every notebook)
1. `!pip install` cell — non-standard packages only (timm, torchviz, etc.)
2. Imports cell
3. Theory/concept markdown cell
4. Code demonstration cells
5. Exercise cell at end (2–3 challenges for students)

### Marking Changes
- New or updated cells must begin with `### [UPDATED]` (in markdown) or `# [UPDATED]` (in code)
- This helps instructors and students identify what changed between versions

### pip install rules
- Always add `!pip install <package>` cells at the top of notebooks that need non-Colab-standard packages
- Non-standard packages requiring install: `timm`, `torchviz`, `torchinfo`, `pycocotools`, `grad-cam`
- Standard Colab packages (no install needed): `torch`, `torchvision`, `numpy`, `matplotlib`, `sklearn`, `tqdm`, `PIL`, `cv2`

### Visualization decisions
- Use **CIFAR10 batch images** (already loaded in relevant notebooks) for feature maps, Grad-CAM, etc.
- No external image downloads unless absolutely necessary
- All plots must use `matplotlib` with `plt.tight_layout()` and clear titles/axis labels
- Use `fig, axes` subplot style for side-by-side comparisons

### Code style
- Device-agnostic: always use `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Use `weights=ModelName_Weights.DEFAULT` — never `pretrained=True` (deprecated since torchvision 0.13)
- Use `model.train()` / `model.eval()` explicitly before training/inference
- Seed reproducibility: `torch.manual_seed(42)` at the top of training cells

---

## Phase Progress Tracker

### Phase 1 — Fix Critical Issues ✅ DONE
- [x] Replace `pretrained=True` → `weights=ModelWeights.DEFAULT` in all notebooks and scripts
- [x] Fix `run_custom_webcam.py` crash (undefined `out.release()`)
- [x] Add `requirements.txt`
- [x] Remove duplicate `11_Batch_Normalization_and_Dropout.ipynb`
- [x] Update all Colab badge URLs in README to point to feature branch

### Phase 2 — Modernize Existing Notebooks ✅ DONE
Notebooks updated: 02, 06, 10, 11, 12, 13, 14, 15, 16, 17

- [x] **02**: Optimizer comparison plot (SGD vs Momentum vs Adam), 3D loss landscape, LR sensitivity
- [x] **06**: LR schedulers (StepLR, CosineAnnealingLR, OneCycleLR), gradient clipping, val curves
- [x] **10**: Feature map visualizations, Grad-CAM, receptive field explanation
- [x] **11**: BatchNorm distribution plots, Dropout train/eval demo, ResidualBlock diagram
- [x] **12**: Model comparison table, `timm` intro, feature extraction visualization
- [x] **13**: Feature extraction vs fine-tuning comparison, `timm` fine-tuning example
- [x] **14**: TorchVision transforms v2, `num_workers`+`pin_memory`, AutoAugment/RandAugment
- [x] **15 & 16**: Latent space interpolation, t-SNE visualization, VAE ELBO theory
- [x] **17**: GAN training stability tips, failure mode examples, FID score explanation

### Phase 3 — New Notebooks (Modern Topics) ✅ DONE
- [x] 18_Mixed_Precision_and_torch_compile.ipynb — FP16/BF16, autocast, GradScaler, torch.compile benchmark
- [x] 19_Modern_Training_Best_Practices.ipynb — gradient accumulation, warmup+cosine, checkpointing, early stopping
- [x] 20_Vision_Transformers_ViT.ipynb — patch embeddings, self-attention, attention maps, ViT fine-tuning
- [x] 21_Diffusion_Models_Intro.ipynb — forward/reverse diffusion, DDPM, DDIM, HuggingFace Diffusers
- [x] 22_CLIP_and_Zero_Shot_Learning.ipynb — contrastive pre-training, zero-shot CIFAR10, prompt engineering
- [x] 24_Experiment_Tracking_WandB.ipynb — wandb.log, wandb.watch, hyperparameter sweeps

### Course Restructuring (done alongside Phase 3)
- [x] Deleted Custom_Dataset_and_Fine_Tuning.ipynb (redundant with NB13 + NB14)
- [x] Removed webcam cells from NB12 (not Colab-compatible)
- [x] Replaced Kaggle API cell in NB17 with HuggingFace Hub download
- [x] Removed Pierian-Data copyright cell from 03_Basic-PyTorch-NN.ipynb
- [x] Rewrote README.md as professional 25-notebook course index with part divisions

### Phase 4 — Cross-Cutting Enhancements ✅ DONE
- [x] Theory block added to NB01, NB04, NB05, NB09, NB24
- [x] `torch.autograd` code demo + gradient flow visualization in NB04
- [x] Worked solution added to NB05 (myModel exercise)
- [x] Output size formula + feature map visualization in NB09
- [x] Worked solutions added to NB08 (California Housing scaffold cells 5–7)
- [x] NB24 (Faster R-CNN) fully modernized: self-contained training loop (no engine.py dependency), theory, dataset download instructions
- [x] "Common Mistakes" callout tables added to NB01, NB04, NB05, NB09, NB24
- [x] "Further Reading" cells added to NB01, NB04, NB05, NB09, NB24
- [x] Exercises section (3–4 challenges) added to NB01, NB04, NB05, NB09, NB24

### Phase 5 — (TBD) ⏳ PENDING

---

## Package Versions (Target)

```
torch>=2.6.0
torchvision>=0.21.0
timm>=1.0.0
torchinfo>=1.8.0
torchviz>=0.0.2
grad-cam>=1.5.0
matplotlib>=3.7.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

---

## What NOT to Do

- Do NOT push to `main` directly (403 protected)
- Do NOT use `pretrained=True` anywhere
- Do NOT remove existing working code — only add to it
- Do NOT add cells that require downloading large external files without a fallback
- Do NOT break Colab compatibility (avoid local filesystem assumptions)
