# AFNDetection 🛡️
### Arabic Fake News Detection Using a Fine-Tuned AraBERT Transformer

> A deep-learning system and deployable web application for classifying Modern Standard Arabic (MSA) news as **real** (حقيقي) or **fake** (مزيف).

---

## Table of Contents
- [Overview](#overview)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training](#training)
- [Running the Web App](#running-the-web-app)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Tech Stack](#tech-stack)
- [Limitations](#limitations)

---

## Overview

AFNDetection fine-tunes **AraBERT v2** (`aubmindlab/bert-base-arabertv2`) on a curated + augmented Arabic news corpus to detect misinformation in Modern Standard Arabic. The model is served through a **Flask REST API** with an Arabic-first, right-to-left (RTL) web interface for real-time prediction.

Key features:
- End-to-end reproducible training pipeline (Google Colab + local)
- Arabic text normalisation (Unicode NFKC, diacritic removal, tatwīl removal)
- Layer-wise learning-rate decay (LLRD), label smoothing, early stopping
- Production-ready Flask app with RTL Arabic UI and confidence visualisation
- YAML-driven configuration — swap models without touching code

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | **0.830** |
| Test F1 (binary) | **0.829** |
| Precision (real / fake) | ~0.83 / ~0.83 |
| Recall (real / fake) | ~0.82 / ~0.84 |
| Test set size | 636 samples (318 real, 318 fake) |
| Best val F1 (epoch 7) | 0.815 |

Confusion matrix: `[[267, 52], [56, 261]]` — errors are symmetric (~17% each class), indicating no systematic bias.

---

## Project Structure

```
AFNDetection/
├── app/
│   ├── flask_app.py               # Flask backend (REST API + model loading)
│   └── web/
│       ├── static/
│       │   ├── css/style.css      # Dark RTL theme, glass-morphism
│       │   └── js/app.js          # Canvas particles, API calls, result rendering
│       └── templates/index.html   # Arabic-first Jinja2 template
├── checkpoints/
│   └── arabic/                    # Saved model (config.json, model.safetensors, ...)
├── configs/
│   ├── app.yaml                   # Checkpoint path for the Flask app
│   ├── train_arabic.yaml          # Full training config
│   └── train_arabic_curated.yaml  # Curated-seed-only training config
├── data/
│   └── raw/
│       └── arabic_curated.csv     # 100 hand-curated balanced samples (seed corpus)
├── docs/
│   └── images/                    # Project logos
├── notebooks/
│   └── AFNDetection_Colab.ipynb   # Google Colab training notebook
├── src/fndarija/
│   ├── data/preprocess.py         # Arabic normalisation pipeline
│   └── inference/predict.py       # Inference utilities
├── pyproject.toml
└── requirements.txt
```

---

## Installation

**Requirements:** Python ≥ 3.10

```bash
# Clone the repository
git clone https://github.com/Zaynab-AitAddi/Project-DEEP-LEARNING-Arabic-Fake-News-Detection.git
cd AFNDetection

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

**Key dependencies:**

| Package | Version |
|---|---|
| torch | ≥ 2.1.0 |
| transformers | ≥ 4.38.0 |
| datasets | ≥ 2.18.0 |
| flask | ≥ 3.0.0 |
| pyarabic | ≥ 0.6.15 |

---

## Training

### Option A — Google Colab (recommended)

1. Open `notebooks/AFNDetection_Colab.ipynb` in Google Colab
2. Set the runtime to **T4 GPU** (Runtime → Change runtime type)
3. Mount Google Drive when prompted
4. Run all cells — checkpoints are saved to your Drive automatically
5. Download `afndetection_checkpoint.zip` when training completes

### Option B — Local

```bash
# Edit configs/train_arabic.yaml to set your data paths, then:
python -m fndarija.train --config configs/train_arabic.yaml
```

**Training hyperparameters (defaults):**

| Parameter | Value |
|---|---|
| Base learning rate | 3e-5 |
| Optimizer | AdamW + LLRD (decay 0.9) |
| Batch size (train / eval) | 16 / 32 |
| Epochs | 8 (early stopping, patience 3) |
| Label smoothing | 0.1 |
| Max sequence length | 256 tokens |
| Warmup ratio | 0.10 |
| LR scheduler | Cosine with warmup |

Training on 5,293 samples takes roughly **45–60 minutes** on a T4 GPU.

---

## Running the Web App

```bash
# 1. Make sure configs/app.yaml points to your checkpoint:
#    arabic_checkpoint: "checkpoints/arabic"

# 2. Start the Flask server
python app/flask_app.py

# 3. Open your browser at:
#    http://localhost:8501
```

**API endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the RTL Arabic web interface |
| `/api/predict` | POST | Accepts `{"text": "..."}`, returns prediction + probabilities |
| `/api/health` | GET | Health check for deployment monitoring |

**Example request:**
```bash
curl -X POST http://localhost:8501/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "أعلنت الوزارة في بلاغ رسمي أن الامتحانات ستنطلق وفق الجدولة المعتمدة"}'
```

**Example response:**
```json
{
  "prediction": "real",
  "prediction_ar": "حقيقي",
  "confidence": 0.92,
  "probabilities": { "fake": 0.08, "real": 0.92 }
}
```

---

## Dataset

The training corpus (5,293 samples) was assembled from three sources:

| Source | Samples | Description |
|---|---|---|
| Curated seed (`arabic_curated.csv`) | 100 | Hand-cleaned, balanced (50 real / 50 fake) |
| Hugging Face Hub (`Nahla-yasmine/arabic_fake_news`) | up to 5,000 | Web-scraped, balanced subset |
| Offline augmentation | ~193 | 2 variants per curated sample (word deletion, swap, diacritic noise) |

**Stratified splits** (seed = 42):

| Split | Ratio | Real | Fake | Total |
|---|---|---|---|---|
| Training | 80% | ~2,647 | ~2,647 | 5,293 |
| Validation | 10% | ~330 | ~330 | ~660 |
| Test | 10% | 318 | 318 | 636 |

**Normalisation pipeline:** Unicode NFKC → strip diacritics → remove tatwīl → collapse whitespace

---

## Model Architecture

```
Arabic News Text
      ↓
AraBERT v2 Tokenizer (BPE, ~64k vocab, max_len=256)
      ↓
AraBERT v2 Encoder (12 transformer layers, 768 hidden, 12 heads, 110M params)
      ↓
[CLS] vector (768-d)
      ↓
Dropout (p=0.1) → Linear (768→2) → Softmax
      ↓
P(fake), P(real)
```

**Base model:** `aubmindlab/bert-base-arabertv2`  
Pre-trained on 70M Arabic sentences (8.5B tokens) from news, Wikipedia, and books.

**Checkpoint artifacts** (saved to `checkpoints/arabic/`):

| File | Description |
|---|---|
| `config.json` | Architecture config, label mapping |
| `model.safetensors` | Model weights (~540 MB) |
| `tokenizer.json` | Vocabulary and tokenisation rules |
| `label_schema.json` | `id2label` / `label2id` mapping |
| `test_metrics.json` | Held-out test performance |
| `optimizer.pt` | Optimizer state (for resuming training) |

---

## Tech Stack

| Layer | Tools |
|---|---|
| NLP core | AraBERT v2, Hugging Face Transformers / Trainer |
| Arabic preprocessing | pyarabic |
| Deep learning framework | PyTorch 2.1+ |
| Training infrastructure | Google Colab Pro, NVIDIA Tesla T4 (16 GB) |
| Deployment | Flask 3.0, HTML/CSS/JS (RTL) |
| Configuration | YAML |

---

## Limitations

- **MSA only** — weak on dialectal Arabic (Darija, Egyptian, Levantine)
- **Binary classification** — no neutral/satire/opinion category
- **No source verification** — stylistic analysis only, no fact-checking
- **Short-text optimised** — best on 50–150 word articles; very short or very long texts may be less accurate
- **Static model** — does not adapt to evolving misinformation without retraining

> ⚠️ This tool is a research and educational aid. It should never be the sole basis for content moderation or publishing decisions. A human must remain in the loop.

---

## License

This project is released for academic and educational use.  
**Author:** Zaynab AITADDI — ENSAH, Université Abdelmalek Essaâdi
