# AI-Medical-and-NST-Project
<!-- banner -->
<p align="center">
  <img src="assets/banner.svg" alt="AI Medical + NST Project" width="900"/>
</p>

# AI Vision — Medical Image Classification & Neural Style Transfer

**Custom VGG19-based medical image classifier (pneumonia & skin cancer)** and **Neural Style Transfer (NST)** feature extractor, wrapped as Flask APIs and served with a modern TailwindCSS frontend.

**Authors:** Tarun Sai Nath, Manish Kumar Das, Purbasha Nayak  
**Guide:** Dr. Nibedan Panda

---

## 🔎 Project Overview

This repository contains two main systems:

1. **Medical Image Classification (Healthcare)**  
   - Custom VGG19-inspired CNN trained from scratch to classify images into:
     - `pneumonia_positive`, `pneumonia_negative`, `skin_cancer_positive`, `skin_cancer_negative`
   - CPU-optimized (GlobalAveragePooling, BatchNorm, Dropout, L2 regularization)
   - Exposes a REST API: `POST /predict` (returns JSON with label + confidence)

2. **Neural Style Transfer (NST)**  
   - Custom VGG19 feature-extractor used to compute content & style representations
   - Performs iterative optimization to produce stylized JPG images
   - Exposes a REST API: `POST /stylize` (accepts `content` + `style` images; returns stylized JPG)

---

## ⭐ Features

- Lightweight, CPU-friendly model architecture
- End-to-end Flask APIs for both services
- Modern glassmorphic frontend (TailwindCSS)
- Simple training and inference scripts
- Usable for demo, experimentation, and academic submission

---

## 📁 Repo Structure (high-level)

├─ Final/ # main code: API servers, inference, utils
│ ├─ medical_service.py # Flask server for /predict
│ ├─ nst_api.py # Flask server for /stylize
│ ├─ inference_new.py # local inference script (medical)
│ ├─ nst_model_generator.py # build + save NST feature extractor
│ └─ ...
├─ Neural-Style_Transfer/ # NST notebooks & helper code
├─ Dataset/ # train/val folders (not included)
├─ Result/ # saved models & labels (keep private or use LFS)
├─ uploads_medical/ # runtime uploads (gitignored)
├─ uploads_nst/ # runtime uploads (gitignored)
├─ nst_output/ # NST outputs (gitignored)
├─ assets/ # README banner & screenshots
├─ index.html # frontend one-page
├─ requirements.txt # pip dependencies
└─ README.md
