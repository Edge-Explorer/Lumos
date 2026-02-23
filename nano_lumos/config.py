import os

# Model Architecture
D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 8
D_FF = 512
MAX_LEN = 128
DROPOUT = 0.1

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DEVICE = "cpu"  # Change to "cuda" if you have an NVIDIA GPU

# Paths
DATA_DIR = "data"
RAW_DATA = "../stories.csv"
TRAIN_DATA = os.path.join(DATA_DIR, "sci_fi_nano.txt")
TOKENIZER_PREFIX = os.path.join(DATA_DIR, "nano_tokenizer")
MODEL_SAVE_PATH = "nano_lumos.pth"

# Tokenizer
VOCAB_SIZE = 8000  # Smaller vocab for smaller model
