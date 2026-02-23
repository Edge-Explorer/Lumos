import pandas as pd
import os
import config

def clean_text(text):
    # Simple cleaning: remove Gutenberg headers/footers if possible, 
    # but for nano-stories, we just want raw text snippets
    text = str(text).replace("\n", " ").replace("\r", " ")
    return " ".join(text.split())

def prepare_data():
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)
        
    print(f"Loading {config.RAW_DATA}...")
    # Using chunksize for memory efficiency since it's a large file
    chunk_size = 10000
    sci_fi_keywords = ["science fiction", "galaxy", "rocket", "planet", "alien", "space", "future", "quantum", "stellar"]
    
    extracted_count = 0
    max_stories = 1000 # Let's start with 1000 good stories for the nano-model
    
    with open(config.TRAIN_DATA, "w", encoding="utf-8") as f:
        for chunk in pd.read_csv(config.RAW_DATA, chunksize=chunk_size):
            # Simple heuristic: if chunk contains sci-fi keywords in the content
            mask = chunk['content'].str.contains('|'.join(sci_fi_keywords), case=False, na=False)
            sci_fi_stories = chunk[mask]
            
            for content in sci_fi_stories['content']:
                cleaned = clean_text(content)
                if len(cleaned) > 500: # Only take reasonably long text
                    f.write(cleaned + "\n")
                    extracted_count += 1
                
                if extracted_count >= max_stories:
                    break
            
            if extracted_count >= max_stories:
                break
                
    print(f"Done! Extracted {extracted_count} Sci-Fi stories to {config.TRAIN_DATA}")

if __name__ == "__main__":
    prepare_data()
