<p align="center">
  <img src="https://img.shields.io/github/stars/avinash064/Avinash?style=social" />
  <img src="https://img.shields.io/github/last-commit/avinash064/Avinash?color=blue" />
  <img src="https://img.shields.io/badge/Built%20With-PyTorch%20%7C%20ViT%20%7C%20timm-blue?style=flat-square" />
</p>

<h1 align="center">🧠 MAE-ViT Foundation Model for Anime Understanding</h1>

<p align="center">
  <b>Self-supervised Vision Transformer Pretraining on Anim400K + AnitaDataset</b>
</p>

---

## 📑 Table of Contents
- [🚀 Overview](#-overview)
- [🧠 Architecture](#-architecture)
- [📦 Datasets](#-datasets)
- [⚙️ Training Pipeline](#️-training-pipeline)
- [🧪 Inference](#-inference)
- [📁 Code Structure](#-code-structure)
- [☁️ Deployment](#-deployment)
- [📊 Results](#-results)
- [🔬 Applications](#-applications)
- [📚 Citation](#-citation)
- [👤 Author](#-author)
- [🚧 Roadmap](#-roadmap)

---

## 🚀 Overview
This project implements a foundation model for anime vision using:
- **Masked Autoencoders (MAE)** + **Vision Transformer (ViT)**
- **Large-scale self-supervised learning**
- **Modular training pipeline**
- Based on: **Anim400K** & **AnitaDataset**

---

## 🧠 Architecture

<details>
<summary><strong>Click to view MAE + ViT pipeline</strong></summary>

```text
Input Image (224x224)
→ Patch Embedding (16x16 patches)
→ ViT Encoder (masked tokens)
→ Lightweight MLP Decoder
→ Reconstructed Image (MSE Loss)
```

- ✅ Encoder: ViT-B/16 (timm pretrained)
- ✅ Decoder: Shallow MLP
- ✅ Loss: Mean squared error (masked pixels only)
</details>

---

## 📦 Datasets

<details>
<summary>🎥 <strong>Anim400K (pretraining)</strong></summary>

```
datasets/anim400k/
├── video_clips/             # MP4 clips in folders
├── frames/                  # Extracted images {video_id}/frame_XXXX.jpg
├── audio_clips/            # Optional .wav files
├── character_pics/         # Reference character images
└── splits.json              # Annotations
```
</details>

<details>
<summary>🎴 <strong>AnitaDataset (fine-tuning)</strong></summary>

```
datasets/anitadataset/
├── images/
├── annotations.json
└── metadata.json
```
</details>

---

## ⚙️ Training Pipeline

### 🧠 MAE Pretraining (on Anim400K)
```bash
python train_mae.py \
  --data_root datasets/anim400k/frames \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-4 \
  --ckpt_dir checkpoints/
```

### 🎯 Fine-tuning (on AnitaDataset)
```bash
python train_anita.py \
  --data_root datasets/anitadataset/images \
  --annotations datasets/anitadataset/annotations.json \
  --pretrained_ckpt checkpoints/mae_epoch_100.pt \
  --epochs 30 \
  --lr 5e-5
```

---

## 🧪 Inference

Use `scripts/infer_reconstruction.py`:
```bash
python scripts/infer_reconstruction.py \
  --model_ckpt checkpoints/mae_epoch_100.pt \
  --image_path datasets/anim400k/frames/1234/frame_0001.jpg
```

Output: Original + Masked + Reconstructed grid

---

## 📁 Code Structure

```
.
├── train_mae.py            # MAE Pretraining
├── train_anita.py          # Finetuning
├── models/                 # MAEWrapper, ViT, Decoder
├── config/                 # YAML configs
├── datasets/               # FrameDataset, Annotations
├── scripts/                # Inference, frame extractor
└── README.md
```

---

## ☁️ Deployment

### ☁️ Upload to Google Cloud
```bash
pip install gcsfs google-cloud-storage

# Upload datasets
gsutil cp -r datasets/anim400k gs://anim-foundation-avinash/
gsutil cp -r datasets/anitadataset gs://anim-foundation-avinash/
```

### 🌐 Push to GitHub
```bash
git init
git remote add origin https://github.com/avinash064/Avinash.git
git add .
git commit -m "Initial commit: MAE Foundation Model"
git push origin main
```

---

## 📊 Results

| Dataset      | MAE Loss ↓ | PSNR ↑ | SSIM ↑ |
| ------------ | ---------- | ------ | ------ |
| Anim400K     | 0.029      | 23.4   | 0.78   |
| AnitaDataset | 0.025      | 24.9   | 0.82   |

---

## 🔬 Applications
- Anime Character Reconstruction
- Facial Expression Synthesis
- Pose Transfer
- Lip Syncing
- Video Super-Resolution

---

## 📚 Citation
```bibtex
@misc{Avinash2025MAEFoundation,
  author = {Avinash Kashyap},
  title = {MAE Pretraining on Anim400K and AnitaDataset},
  year = 2025,
  howpublished = {\url{https://github.com/avinash064/Avinash}},
}
```

---

## 👤 Author
**Avinash Kashyap**  
🎓 AI & Medical Imaging | 🔬 Deep Learning | 🚀 Foundation Models  
🔗 [GitHub](https://github.com/avinash064) · [LinkedIn](https://linkedin.com/in/avinash-kashyap-7b29b4247)

---

## 🚧 Roadmap

- [x] MAE Pretraining (Anim400K)
- [x] Finetuning (AnitaDataset)
- [x] Cloud Upload (GCP Bucket)
- [x] GitHub Project Integration
- [ ] Add Audio + Character Fusion
- [ ] Add HuggingFace Model Card
- [ ] Publish Paper + Demo Site

---

> ❤️ Star this repo and tag [@avinash064](https://github.com/avinash064) if you use this project!
