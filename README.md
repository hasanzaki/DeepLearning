# MCTA 4363 — Deep Learning

> **Framework:** PyTorch ≥ 2.6 &nbsp;|&nbsp; **Platform:** Google Colab (GPU) &nbsp;|&nbsp; **Level:** Undergraduate

A hands-on deep learning course covering foundational theory through modern architectures — from linear classifiers to diffusion models and CLIP. Every notebook runs on Google Colab with a single click.

---

## Prerequisites

| Topic | What you need |
|-------|--------------|
| Programming | Python 3 — functions, classes, list comprehensions |
| Mathematics | Linear algebra (vectors, matrices, dot products), calculus (derivatives, chain rule) |
| Tools | A Google account to run notebooks on Colab |

---

## Quick Start

Click any **Open in Colab** badge below. No local setup required — all notebooks run on free GPU (Tesla T4).

For local development:
```bash
git clone https://github.com/hasanzaki/DeepLearning.git
cd DeepLearning
pip install -r requirements.txt
jupyter notebook
```

---

## Course Curriculum

### Part 1 — Foundations

| # | Notebook | Key Concepts | Colab |
|---|----------|-------------|-------|
| 01 | Introduction & Linear Classifiers | Perceptron, SVM, loss functions, decision boundaries | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/01_Introduction_and_Linear_Classifiers.ipynb) |
| 02 | Optimization & Gradient Descent | SGD, momentum, Adam, loss landscapes, LR sensitivity | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/02_Optimization_Gradient_Descent.ipynb) |
| 03 | Neural Networks — Theory | MLPs, activation functions, universal approximation theorem | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/03_Neural_networks.ipynb) |
| 04 | Backpropagation | Chain rule, computational graphs, vanishing gradients | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/04_Backpropagation.ipynb) |
| 05 | Basic PyTorch Neural Networks | Tensors, `nn.Module`, training loop, model saving | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/03_Basic-PyTorch-NN.ipynb) |
| 06 | Build a Neural Network in PyTorch | DataLoaders, custom architectures, regularisation | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/05_Build_a_Neural_Network_With_Pytorch.ipynb) |

---

### Part 2 — Training Fundamentals

| # | Notebook | Key Concepts | Colab |
|---|----------|-------------|-------|
| 07 | Train a FFNN | Full training loop, LR schedulers, gradient clipping, val curves | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/06_Train_a_FFNN_new.ipynb) |
| 08 | FFNN — Regression | California Housing, MSE loss, feature scaling (exercise notebook) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/07_FFNN_CaliforniaHousing.ipynb) |

---

### Part 3 — Convolutional Neural Networks

| # | Notebook | Key Concepts | Colab |
|---|----------|-------------|-------|
| 09 | Principles of Convolution | Kernels, stride, padding, feature maps | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/08.%20Principles%20of%20the%20Convolution.ipynb) |
| 10 | Convolution in Practice | `Conv2d`, pooling, spatial dimensions, receptive field | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/09.%20Convolution_in_practice.ipynb) |
| 11 | Build a CNN | CIFAR10 CNN, feature map visualisation, Grad-CAM | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/10.%20Build_a_CNN.ipynb) |
| 12 | BatchNorm, Dropout & Skip Connections | Distribution plots, train/eval modes, ResidualBlock | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/11_Batch_Normalization_Dropout_SkipConnection.ipynb) |

---

### Part 4 — Modern Training Techniques ✦ New

| # | Notebook | Key Concepts | Colab |
|---|----------|-------------|-------|
| 13 | Mixed Precision & torch.compile | FP16/BF16, `autocast`, `GradScaler`, `torch.compile`, benchmarking | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/18_Mixed_Precision_and_torch_compile.ipynb) |
| 14 | Modern Training Best Practices | Gradient accumulation, LR warmup, checkpointing, early stopping | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/19_Modern_Training_Best_Practices.ipynb) |

---

### Part 5 — Pretrained Models & Transfer Learning

| # | Notebook | Key Concepts | Colab |
|---|----------|-------------|-------|
| 15 | Pretrained Models | AlexNet, ResNet, EfficientNet, `timm`, model comparison, PCA embeddings | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/12_Image_Classification_using_pre_trained_models.ipynb) |
| 16 | Transfer Learning | Fine-tuning vs feature extraction, layer freezing, `timm` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/13.%20Transfer_Learning.ipynb) |
| 17 | Custom Dataset & Data Augmentation | `ImageFolder`, transforms v2, AutoAugment, RandAugment | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/14.%20Custom_Dataset_and%20Data_Augmentation.ipynb) |
| 18 | Vision Transformers (ViT) ✦ | Patch embeddings, self-attention, attention maps, ViT fine-tuning | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/20_Vision_Transformers_ViT.ipynb) |

---

### Part 6 — Generative Models

| # | Notebook | Key Concepts | Colab |
|---|----------|-------------|-------|
| 19 | Autoencoders | Encoder-decoder, latent space, interpolation, t-SNE | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/15.%20Autoencoders.ipynb) |
| 20 | Variational Autoencoders (VAE) | ELBO, reparameterisation trick, KL divergence, latent interpolation | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/16.%20VAE.ipynb) |
| 21 | Generative Adversarial Networks | DCGAN, training stability, FID score, failure modes | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/17.%20GAN_anime.ipynb) |
| 22 | Diffusion Models ✦ | Forward/reverse diffusion, DDPM, DDIM, HuggingFace Diffusers | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/21_Diffusion_Models_Intro.ipynb) |

---

### Part 7 — Multimodal AI ✦ New

| # | Notebook | Key Concepts | Colab |
|---|----------|-------------|-------|
| 23 | CLIP & Zero-Shot Learning ✦ | Contrastive pre-training, image-text similarity, zero-shot classification, prompt engineering | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/22_CLIP_and_Zero_Shot_Learning.ipynb) |

---

### Part 8 — Applied & Tooling

| # | Notebook | Key Concepts | Colab |
|---|----------|-------------|-------|
| 24 | Object Detection — Faster R-CNN | Region proposals, FPN, COCO format, custom detection | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/test_RCNN.ipynb) |
| 25 | Experiment Tracking with W&B ✦ | `wandb.log`, `wandb.watch`, hyperparameter sweeps, dashboards | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasanzaki/DeepLearning/blob/claude/analyze-github-repo-Pa8D0/24_Experiment_Tracking_WandB.ipynb) |

> **✦** = New notebook added in 2025 course update

---

## Key Dependencies

```
torch>=2.6.0          torchvision>=0.21.0     timm>=1.0.0
diffusers>=0.27.0     transformers>=4.40.0    openai-clip
wandb>=0.17.0         torchinfo>=1.8.0        scikit-learn>=1.3.0
```

Full list: [`requirements.txt`](requirements.txt)

---

## Repository Structure

```
DeepLearning/
├── data/                    Sample datasets (Iris)
├── 01_Introduction...       Course notebooks (01–25)
├── run_pretrained_image.py  Static image classification script
├── run_pretrained_webcam.py Live webcam classification script
├── run_custom_webcam.py     Custom model webcam inference
├── requirements.txt         Python dependencies
└── CLAUDE.md                AI assistant project governance
```

---

## License

Course materials are for educational use within MCTA 4363.
PyTorch is distributed under the BSD licence. Individual pretrained model weights are subject to their respective licences (see HuggingFace Hub and TorchVision documentation).
