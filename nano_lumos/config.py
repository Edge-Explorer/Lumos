import os
import torch

# ── Device ──────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model Architecture (Upgraded for T4 GPU) ─────────────────────────────────
D_MODEL    = 512          # was 256 — 4x more "thinking" capacity
NUM_LAYERS = 6            # was 4
NUM_HEADS  = 8
D_FF       = 2048         # was 512 — wider feed-forward
MAX_LEN    = 256          # was 128 — longer context window
DROPOUT    = 0.1

# ── Training Hyperparameters ─────────────────────────────────────────────────
BATCH_SIZE     = 64       # was 16 — keeps the T4 GPU busy
LEARNING_RATE  = 3e-4     # slightly higher for faster learning
NUM_EPOCHS     = 50       # more epochs with GPU power
GRAD_CLIP      = 1.0      # prevents exploding gradients

# ── DataLoader ───────────────────────────────────────────────────────────────
NUM_WORKERS    = 2        # parallel CPU workers to feed the GPU

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
RAW_DATA        = "stories.csv"
TRAIN_DATA      = os.path.join(DATA_DIR, "sci_fi_nano.txt")
TOKENIZER_PREFIX = os.path.join(DATA_DIR, "nano_tokenizer")
MODEL_SAVE_PATH  = "lumos_v3.pth"

# ── Tokenizer ────────────────────────────────────────────────────────────────
VOCAB_SIZE = 8000          # back to 8000 now that we have 10k+ stories
