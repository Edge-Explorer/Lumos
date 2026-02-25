import os
import re
import sys
sys.path.append(os.path.dirname(__file__))
import config

def clean_text(text):
    """Clean story text: remove formatting noise, collapse whitespace."""
    text = str(text)
    # Remove Reddit formatting artifacts like "**bold**", "&gt;", "&amp;"
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)   # **bold** / *italic*
    text = re.sub(r'&gt;', '', text)                       # quoted lines
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)              # [links](url)
    text = re.sub(r'http\S+', '', text)                     # bare URLs
    text = re.sub(r'\n{3,}', '\n\n', text)                 # excessive newlines
    # Keep only standard English characters + punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'";\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_data(max_stories=10000):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)

    print("Downloading WritingPrompts dataset from HuggingFace...")
    # euclaise/writingprompts: columns = 'prompt', 'story'
    ds = load_dataset("euclaise/writingprompts", split="train")
    print(f"Total stories available: {len(ds)}")

    sci_fi_keywords = [
        "space", "planet", "alien", "robot", "future", "quantum",
        "galaxy", "starship", "time travel", "dimension", "ai",
        "android", "laser", "warp", "nebula", "orbit", "cosmos",
        "mutation", "radiation", "cyborg", "singularity", "dystopia",
        "apocalypse", "interstellar", "colony", "simulation",
    ]

    extracted = 0
    with open(config.TRAIN_DATA, "w", encoding="utf-8") as f:
        for item in ds:
            prompt = item.get("prompt", "").strip()
            story  = item.get("story",  "").strip()

            # Only keep Sci-Fi / speculative stories
            combined = (prompt + " " + story).lower()
            if not any(kw in combined for kw in sci_fi_keywords):
                continue

            cleaned_story  = clean_text(story)
            cleaned_prompt = clean_text(prompt)

            # Reject very short stories
            if len(cleaned_story.split()) < 100:
                continue

            # Format: "[PROMPT] <prompt> [STORY] <story>"
            line = f"[PROMPT] {cleaned_prompt} [STORY] {cleaned_story}"
            f.write(line + "\n")
            extracted += 1
            print(f"  Extracted {extracted}/{max_stories}...", end="\r")

            if extracted >= max_stories:
                break

    print(f"\nDone! {extracted} clean Sci-Fi stories saved to {config.TRAIN_DATA}")

if __name__ == "__main__":
    prepare_data()
