# ✨ Lumos — A Transformer Built From Scratch

> *"Not imported. Not fine-tuned. Built from the ground up."*

Lumos is a custom **GPT-style Transformer language model** built entirely from scratch in **PyTorch** — no Hugging Face models, no shortcuts. Every layer, every attention head, every tensor operation was hand-coded to deeply understand the architecture powering today's most advanced AI systems.

---

## 🧠 What Is Lumos?

Lumos is a personal deep learning research project that explores the internals of the Transformer architecture, trained on a curated **Sci-Fi** text corpus (Reddit WritingPrompts) to generate science fiction story premises.

The project has evolved through 3 versions:

| Version | Name | Description |
|---|---|---|
| V1 | `legacy_lumos` | Full-scale Encoder-Decoder Transformer, trained on a 1GB Gutenberg corpus |
| V2 | `nano_lumos` (CPU) | GPT-style Decoder-only Transformer — first generation on local CPU |
| **V3** | `nano_lumos` (GPU) | **Current** — 27M params, 10k clean stories, T4 GPU, final loss ~2.5 |

---

## 🏗️ Architecture (Built From Scratch)

```
Input Tokens
     │
     ▼
Token Embedding + Positional Encoding
     │
     ▼
┌────────────────────┐
│   Decoder Block ×6 │
│  ┌──────────────┐  │
│  │ Masked Multi │  │
│  │ Head Attention│  │
│  └──────────────┘  │
│  ┌──────────────┐  │
│  │ Feed Forward │  │
│  │   Network    │  │
│  └──────────────┘  │
└────────────────────┘
     │
     ▼
Linear + Softmax → Next Token
```

### Key Components Implemented:
- ✅ **Scaled Dot-Product Attention** — The mathematical heart of every modern AI model
- ✅ **Multi-Head Attention** — Parallel attention across 8 representation subspaces
- ✅ **Positional Encoding** — Sinusoidal embeddings so the model understands word order
- ✅ **Causal (Masked) Self-Attention** — Prevents the model from "cheating" by looking at future tokens
- ✅ **Position-wise Feed Forward Network** — Non-linear transformation at every layer
- ✅ **Layer Normalization + Residual Connections** — Stable training via skip connections
- ✅ **Custom BPE Tokenizer** — Trained from scratch using SentencePiece on 10k Sci-Fi stories
- ✅ **AdamW + Cosine Annealing LR** — Production-grade optimizer setup
- ✅ **Gradient Clipping** — Prevents exploding gradients during training

---

## 🎯 Sample Outputs (Lumos V3)

After 50 epochs on T4 GPU, trained on 10,000 WritingPrompts Sci-Fi stories:

```
Prompt: "The first human on Mars discovered"
Result: "The first human on Mars discovered a new system of Earth orbits."

Prompt: "A robot woke up alone on an alien planet"
Result: "A robot woke up alone on an alien planet ... I was alone i..."

Prompt: "In the year 3000 humanity had finally"
Result: "In the year 3000 humanity had finally been wiped out."
```

> From completely random noise at Epoch 1 → coherent Sci-Fi sentences at Epoch 50. 🚀

---

## 📁 Project Structure

```
Lumos/
│
├── nano_lumos/              # 🚀 Active: Lumos V3
│   ├── config.py            # All model & training settings
│   ├── model.py             # Transformer architecture from scratch
│   ├── data_prep.py         # Downloads & cleans WritingPrompts from HuggingFace
│   ├── train.py             # Tokenizer + training loop (GPU-optimized)
│   └── generate.py          # Text generation with temperature sampling
│
├── legacy_lumos/            # 📦 Archive: V1 & V2 checkpoints
│   ├── Lumos_test.ipynb     # Original V1 encoder-decoder notebook
│   ├── lumos_tokenizer.*    # V1 tokenizer
│   ├── lumos_transformer_epoch1.pth  # V1 model weights
│   └── nano_lumos_colab_v1.pth      # V2 model weights
│
├── lumos_v3.pth             # ✅ Latest trained model (27M params)
├── Colab_Lumos.ipynb        # Google Colab training notebook (T4 GPU)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/Edge-Explorer/Lumos.git
cd Lumos
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download & prepare the data (auto-downloads from HuggingFace)
```bash
python nano_lumos/data_prep.py
```

### 5. Train the model
```bash
python nano_lumos/train.py
```

### 6. Generate Sci-Fi text
```bash
python nano_lumos/generate.py "The last starship on Earth discovered"
```

### 7. Optional: Train on Google Colab (Free T4 GPU)
- Upload `Colab_Lumos.ipynb` to [Google Colab](https://colab.research.google.com)
- Set `Runtime → Change runtime type → T4 GPU`
- Run all cells — no manual file uploads needed!

---

## ⚙️ Lumos V3 Model Config

| Hyperparameter | V2 (CPU) | **V3 (T4 GPU)** |
|---|---|---|
| `d_model` | 256 | **512** |
| `num_layers` | 4 | **6** |
| `num_heads` | 8 | **8** |
| `d_ff` | 512 | **2048** |
| `max_len` | 128 | **256** |
| `vocab_size` | 3,000 | **8,000** |
| `batch_size` | 16 | **64** |
| `epochs` | 30 | **50** |
| `optimizer` | Adam | **AdamW + Cosine LR** |
| `training data` | 1k Gutenberg stories | **10k WritingPrompts** |
| `hardware` | CPU | **Tesla T4 GPU** |
| `parameters` | ~3M | **27,114,304** |

---

## 📊 Training Progress (V3 — T4 GPU)

| Epoch | Avg Loss | LR |
|---|---|---|
| 1 | 6.0719 | 3.00e-04 |
| 10 | ~4.18 | 2.60e-04 |
| 20 | 3.4552 | 1.96e-04 |
| 30 | 2.9407 | 1.04e-04 |
| 40 | ~2.65 | 4.73e-05 |
| 50 | **~2.5** ✅ | ~1e-05 |

*Loss dropped from 6.07 → ~2.5 over 50 epochs with clean data and GPU acceleration.*

---

## 🎯 Why Build From Scratch?

Most AI engineers use pre-built libraries. Building from scratch means:

- You **truly understand** why attention works, not just that it works
- You can **debug at the tensor level** — no black boxes
- You learn the **exact tradeoffs** between model size, vocab size, and training data
- It's the difference between **driving a car** and **building an engine**

---

## 🗺️ Roadmap

- [x] Build Transformer architecture from scratch
- [x] Train custom BPE tokenizer
- [x] Train V1 (Encoder-Decoder) on 1GB corpus
- [x] Build GPT-style Decoder-only model
- [x] Clean data pipeline with WritingPrompts dataset
- [x] GPU training on Google Colab T4
- [x] 27M parameter model with AdamW + Cosine LR
- [x] Coherent Sci-Fi sentence generation
- [ ] Add top-p (nucleus) sampling for better diversity
- [ ] Build a simple web UI for text generation
- [ ] Fine-tune on a specific Sci-Fi subgenre

---

## 👤 Author

**Karan Shelar**  
[Portfolio](https://karan-portfolio-opal.vercel.app/) • [LinkedIn](https://in/karan-shelar-779381343) • [GitHub](https://github.com/Edge-Explorer)

---

## 📄 License

MIT License — See `LICENSE` for details.

---

<p align="center">
  <i>Built with curiosity, PyTorch, and a T4 GPU.</i>
</p>
