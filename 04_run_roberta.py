# This script reads all the cleaned reviews.
# For each review, the RoBERTa model returns a score between 0 and 1.
# The score is the probability that the text was written by an AI.
# This score is the main independent variable of the study.

import time
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Configuration of Hyperparameters and paths
MODEL_NAME = "Hello-SimpleAI/chatgpt-detector-roberta" # Model trained to tell human
SAMPLE_SIZE = None        # None means that we use ALL the reviews, not a sample
BATCH_SIZE = 32           # Number of reviews processed at the same time     
MAX_LEN = 256             # Review longer than 256 tokens are cut
CHECKPOINT_EVERY = 5000   # Save the progress every 5 000 reviews
RANDOM_STATE = 42         # Fixed seed, so the run can be reproduced

PROJECT_ROOT = Path.home() / "Desktop" / "Dissertation_Analysis"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_PATH = DATA_DIR / "ai_scores_checkpoint.parquet"
FINAL_PATH = DATA_DIR / "reviews_with_ai_scores.parquet"


# Detects the best available laptop device
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon GPU (MPS)"
    if torch.cuda.is_available():
        return torch.device("cuda"), "NVIDIA GPU (CUDA)"
    return torch.device("cpu"), "CPU"
# Pick the fastest hardware available: MPS on Apple Silicon GPU

# Load the cleaned reviews from disk.
# It SAMPLE_SIZE is set, keep a smaller sample that respects the share of each star rating 
def load_or_init_dataset():
    df = pd.read_parquet(DATA_DIR / "reviews_clean.parquet") # 
    print(f"  Loaded clean dataset: {len(df):,} reviews")
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df):
        print(f"  Stratified sampling: keeping {SAMPLE_SIZE:,} reviews "
              f"balanced by rating")
        df = (
            df.groupby("rating", group_keys=False)
              .apply(lambda x: x.sample(
                  n=int(SAMPLE_SIZE * len(x) / len(df)),
                  random_state=RANDOM_STATE
              ))
              .reset_index(drop=True)
        )
        print(f"  After stratification: {len(df):,} reviews")

    return df

# Allow the script to continue if it stopped before the end.
# If no checkpoint exists, start from zero with empty scores.
# If a checkpoint exists, load the scores already computed and restart from there.
def resume_from_checkpoint(df):
    if not CHECKPOINT_PATH.exists():
        df["AI_generated_score"] = np.nan
        return df, 0

    print(f"  Found checkpoint: {CHECKPOINT_PATH}")
    df_ckpt = pd.read_parquet(CHECKPOINT_PATH)

    # Assumes identical ordering
    if len(df_ckpt) != len(df):
        print("  WARNING: checkpoint size mismatch. Starting from scratch.")
        df["AI_generated_score"] = np.nan
        return df, 0

    df["AI_generated_score"] = df_ckpt["AI_generated_score"].values
    n_done = df["AI_generated_score"].notna().sum()
    print(f"  Resuming from review #{n_done:,} / {len(df):,}")
    return df, n_done


def main():
    print("=" * 60)
    print("RoBERTa AI-detection inference")
    print("=" * 60)

    # --- Device setup ---
    device, device_name = get_device()
    print(f"\nDevice: {device_name}")
    print(f"PyTorch: {torch.__version__}")

    # --- Load Model ---
    # Tokenizer: turns text into numbers the model can read.
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # Tokenizer -> converts text to numerical tokens
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval() # to activate ready-only inference mode
    print("Model ready.")

    # Safety check: make sure class index 1 is really the "AI" (ChatGPT) class.
    print("Label mapping:", model.config.id2label)
    assert model.config.id2label[1] == "ChatGPT", "Model class mapping is not as expected : index 1 is NOT the 'ChatGPT' class - check the model !"

    # --- Dataset ---
    print("\nLoading dataset")
    df = load_or_init_dataset()
    df, start_idx = resume_from_checkpoint(df)

    if start_idx >= len(df):
        print("\nAll reviews already processed. Saving final file.")
        df.to_parquet(FINAL_PATH, index=False)
        print(f"Final file: {FINAL_PATH}")
        return

    # --- Starting Inference ---
    print(f"\nStarting inference from index {start_idx:,}")
    print(f"  Batch size: {BATCH_SIZE} | Max length: {MAX_LEN} tokens")
    print(f"  Checkpoint every: {CHECKPOINT_EVERY:,} reviews")

    # Extract text to Python list for performance
    texts = df["text"].fillna("").tolist()
    scores = df["AI_generated_score"].tolist() # 'AI_generated_score' -> new empty list to store scores as they are generated

    start_time = time.time()
    last_checkpoint = start_idx

# Core loop => Process batches (per pack of 32) and checkpoint periodically reviews => goal : to minimise the risk of loss in the event of an interruption
    with torch.no_grad():
        for i in range(start_idx, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]         # to transform the text into numerical tokens
            enc = tokenizer(
                batch,
                truncation=True,                    # to reduce too long texts
                padding=True,                       # to add neutral tokens to short reviews (so that all reviews have the same length)
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)

            # Go through the reviews in small batches of 32.
            # Turn the text into tokens (cut if too long, pad if too short).
            # Run the model and keep only the probability of the "AI" class.
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]  # This new probability = future 'IA_generated_score' -> verified via model.config.id2label
            batch_scores = probs.cpu().numpy().tolist()

            for j, s in enumerate(batch_scores):
                scores[i + j] = s

            # --- Checkpoint periodically ---
            # Every 5,000 reviews, save the whole dataframe to disk.
            # This way we do not lose our work if the script stops.
            # Also print the speed and the estimated time left.
            done = i + len(batch)
            if done - last_checkpoint >= CHECKPOINT_EVERY:
                df["AI_generated_score"] = scores
                df.to_parquet(CHECKPOINT_PATH, index=False)
                last_checkpoint = done

                elapsed = time.time() - start_time
                processed_this_run = done - start_idx
                speed = processed_this_run / elapsed if elapsed > 0 else 0
                remaining = (len(texts) - done) / speed if speed > 0 else 0

                print(f"  [{done:,}/{len(texts):,}] "
                      f"speed={speed:.1f} rev/s | "
                      f"ETA={remaining/60:.1f} min | "
                      f"checkpoint saved")

    # --- Final save ---
    # Create final file with the new 'AI_generated_score' column for all 700 808 reviews
    # Print a short summary and the score distribution.
    df["AI_generated_score"] = scores
    df.to_parquet(FINAL_PATH, index=False)

    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 60)
    print(f"DONE. Total inference time this run: {total_time:.1f} min")
    print(f"Final dataset: {FINAL_PATH}")
    print(f"Reviews scored: {df['AI_generated_score'].notna().sum():,}")
    print("=" * 60)

    print("\nAI_generated_score distribution:")
    print(df["AI_generated_score"].describe())


if __name__ == "__main__":
    main()