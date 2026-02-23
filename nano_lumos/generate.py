import torch
import sentencepiece as spm
import config
from model import NanoTransformer

def generate(prompt, max_new_tokens=50, temperature=0.7):
    # Load Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(config.TOKENIZER_PREFIX + ".model")
    
    # Load Model
    model = NanoTransformer(
        vocab_size=config.VOCAB_SIZE,
        d_model=config.D_MODEL,
        num_heads=config.NUM_HEADS,
        d_ff=config.D_FF,
        num_layers=config.NUM_LAYERS,
        max_len=config.MAX_LEN
    ).to(config.DEVICE)
    
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    model.eval()
    
    # Tokenize input
    input_ids = sp.encode(prompt, out_type=int)
    input_tensor = torch.tensor([input_ids]).to(config.DEVICE)
    
    print(f"\nPrompt: {prompt}")
    print("Generating...", end="", flush=True)
    
    for _ in range(max_new_tokens):
        if input_tensor.size(1) >= config.MAX_LEN:
            break
            
        with torch.no_grad():
            logits = model(input_tensor)
            
        # Get last token logits
        last_token_logits = logits[0, -1, :] / temperature
        probs = torch.softmax(last_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)
        
        # Stop if EOS
        if next_token.item() == 3:
            break
            
    decoded = sp.decode(input_tensor[0].tolist())
    print(f"\nResult: {decoded}")

if __name__ == "__main__":
    import sys
    prompt = "In the year 3000, humans discovered" if len(sys.argv) < 2 else sys.argv[1]
    generate(prompt)
