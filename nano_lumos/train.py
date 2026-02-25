import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import os
import sys
sys.path.append(os.path.dirname(__file__))
import config
from model import NanoTransformer

class NanoDataset(Dataset):
    def __init__(self, file_path, sp):
        self.sp  = sp
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(line)
        print(f"Dataset loaded: {len(self.data)} stories")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.sp.encode(self.data[idx], out_type=int)
        if len(tokens) > config.MAX_LEN:
            tokens = tokens[:config.MAX_LEN]
        else:
            tokens = tokens + [0] * (config.MAX_LEN - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

def train_tokenizer():
    print("Training tokenizer...")
    spm.SentencePieceTrainer.train(
        input=config.TRAIN_DATA,
        model_prefix=config.TOKENIZER_PREFIX,
        vocab_size=config.VOCAB_SIZE,
        model_type='bpe',
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        character_coverage=0.9995,
    )
    print(f"Tokenizer saved → {config.TOKENIZER_PREFIX}.model")

def train_model():
    print(f"Device: {config.DEVICE.upper()}")
    if config.DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU RAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    sp = spm.SentencePieceProcessor()
    sp.load(config.TOKENIZER_PREFIX + ".model")

    # ── Dataset & DataLoader ───────────────────────────────────────────────
    dataset    = NanoDataset(config.TRAIN_DATA, sp)
    dataloader = DataLoader(
        dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        num_workers = config.NUM_WORKERS,   # parallel loading
        pin_memory  = (config.DEVICE == "cuda"),  # faster GPU transfer
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = NanoTransformer(
        vocab_size = config.VOCAB_SIZE,
        d_model    = config.D_MODEL,
        num_heads  = config.NUM_HEADS,
        d_ff       = config.D_FF,
        num_layers = config.NUM_LAYERS,
        max_len    = config.MAX_LEN,
        dropout    = config.DROPOUT,
    ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr           = config.LEARNING_RATE,
        weight_decay = 0.01,   # regularization
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    start_epoch    = 0
    checkpoint_path = config.MODEL_SAVE_PATH + ".ckpt"

    # ── Resume checkpoint if available ────────────────────────────────────
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint…")
        ckpt = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"]
        print(f"  Resuming from epoch {start_epoch + 1}")

    # ── Training Loop ──────────────────────────────────────────────────────
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs…\n")
    model.train()
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        total_loss  = 0
        num_batches = 0

        for batch in dataloader:
            batch   = batch.to(config.DEVICE, non_blocking=True)
            inputs  = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()
            outputs = model(inputs)

            loss = criterion(
                outputs.reshape(-1, config.VOCAB_SIZE),
                targets.reshape(-1),
            )
            loss.backward()

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

            optimizer.step()
            total_loss  += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = total_loss / num_batches
        lr_now   = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:02d}/{config.NUM_EPOCHS} | "
              f"Loss: {avg_loss:.4f} | LR: {lr_now:.2e}")

        # ── Save checkpoint after every epoch ─────────────────────────────
        torch.save({
            "epoch"          : epoch + 1,
            "model_state"    : model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss"           : avg_loss,
        }, checkpoint_path)

    # ── Final save ────────────────────────────────────────────────────────
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    print(f"\nModel saved → {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    if not os.path.exists(config.TOKENIZER_PREFIX + ".model"):
        train_tokenizer()
    train_model()
