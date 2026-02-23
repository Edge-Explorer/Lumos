# ✨ Lumos — A Transformer Built From Scratch

> *"Not imported. Not fine-tuned. Built from the ground up."*

Lumos is a custom **Transformer language model** built entirely from scratch in **PyTorch** — no Hugging Face, no shortcuts. Every layer, every attention head, every tensor operation was hand-coded to deeply understand the architecture powering today's most advanced AI systems.

---

## 🧠 What Is Lumos?

Lumos is a personal deep learning research project that explores the internals of the Transformer architecture, trained on a curated **Sci-Fi** text corpus to generate science fiction story premises.

The project has two phases:

| Phase | Name | Description |
|---|---|---|
| V1 | `legacy_lumos` | Full-scale Encoder-Decoder Transformer (60M+ params), trained on a 1GB corpus |
| V2 | `nano_lumos` | GPT-style Decoder-only Transformer, smaller and more focused on Sci-Fi generation |

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
│   Decoder Block ×N │
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
- ✅ **Multi-Head Attention** — Parallel attention across different representation subspaces
- ✅ **Positional Encoding** — Sinusoidal embeddings so the model understands word order
- ✅ **Causal (Masked) Self-Attention** — Prevents the model from "cheating" by looking at future tokens
- ✅ **Position-wise Feed Forward Network** — Non-linear transformation at every layer
- ✅ **Layer Normalization + Residual Connections** — Stable training via skip connections
- ✅ **Custom BPE Tokenizer** — Trained from scratch using SentencePiece on the Sci-Fi corpus

---

## 📁 Project Structure

```
Lumos/
│
├── nano_lumos/              # 🚀 Active: Nano Sci-Fi model
│   ├── config.py            # All model & training settings
│   ├── model.py             # Transformer architecture from scratch
│   ├── data_prep.py         # Extracts & cleans Sci-Fi text from raw CSV
│   ├── train.py             # Tokenizer training + model training loop
│   └── generate.py          # Text generation with temperature sampling
│
├── legacy_lumos/            # 📦 Archive: Original V1 Transformer
│   ├── Lumos_test.ipynb     # Full build notebook (encoder-decoder)
│   ├── lumos_tokenizer.*    # Original trained tokenizer
│   └── lumos_transformer_epoch1.pth  # Saved V1 model weights
│
├── requirements.txt         # All dependencies
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

### 4. Prepare the data
```bash
python nano_lumos/data_prep.py
```

### 5. Train the model
```bash
python nano_lumos/train.py
```

### 6. Generate Sci-Fi text
```bash
python nano_lumos/generate.py "In the year 3000, humans discovered"
```

---

## ⚙️ Nano-Lumos Model Config

| Hyperparameter | Value |
|---|---|
| `d_model` | 256 |
| `num_layers` | 4 |
| `num_heads` | 8 |
| `d_ff` | 512 |
| `max_len` | 128 |
| `vocab_size` | 3,000 |
| `batch_size` | 16 |
| `epochs` | 10 |
| `optimizer` | Adam (lr=1e-4) |
| `hardware` | CPU |

---

## 📊 Training Progress

| Epoch | Avg Loss |
|---|---|
| 1 | *In Progress...* |

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
- [x] Build Nano GPT-style model
- [ ] Train Nano model to convergence
- [ ] Add temperature & top-k sampling controls
- [ ] Build a simple web UI for text generation
- [ ] Scale up with GPU training (Google Colab)

---

## 👤 Author

**Karan Shelar**  
[Portfolio](https://karan-portfolio-opal.vercel.app/) • [LinkedIn](https://in/karan-shelar-779381343) • [GitHub](https://github.com/Edge-Explorer)

---

## 📄 License

MIT License — See `LICENSE` for details.

---

<p align="center">
  <i>Built with curiosity, PyTorch, and a lot of patience.</i>
</p>
