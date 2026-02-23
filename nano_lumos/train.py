import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import os
import config
from model import NanoTransformer

class NanoDataset(Dataset):
    def __init__(self, file_path, sp):
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(line.strip())
        self.sp = sp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.sp.encode(text, out_type=int)
        
        # Slicing/Padding
        if len(tokens) > config.MAX_LEN:
            tokens = tokens[:config.MAX_LEN]
        else:
            tokens = tokens + [0] * (config.MAX_LEN - len(tokens))
            
        return torch.tensor(tokens)

def train_tokenizer():
    print("Training tokenizer...")
    spm.SentencePieceTrainer.train(
        input=config.TRAIN_DATA,
        model_prefix=config.TOKENIZER_PREFIX,
        vocab_size=config.VOCAB_SIZE,
        model_type='bpe',
        pad_id=0, unk_id=1, bos_id=2, eos_id=3
    )
    print(f"Tokenizer saved to {config.TOKENIZER_PREFIX}.model")

def train_model():
    # Load Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(config.TOKENIZER_PREFIX + ".model")
    
    # Dataset & Dataloader
    dataset = NanoDataset(config.TRAIN_DATA, sp)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    # Model
    model = NanoTransformer(
        vocab_size=config.VOCAB_SIZE,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        num_layers=config.NUM_LAYERS,
        max_len=config.MAX_LEN,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    print(f"Starting training on {config.DEVICE}...")
    model.train()
    for epoch in range(config.NUM_EPOCHS):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(config.DEVICE)
            
            # For GPT-style: input is tokens[:-1], target is tokens[1:]
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs.view(-1, config.VOCAB_SIZE), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    if not os.path.exists(config.TOKENIZER_PREFIX + ".model"):
        train_tokenizer()
    train_model()
