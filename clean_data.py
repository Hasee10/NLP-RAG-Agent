import pandas as pd
import numpy as np
import re
import os
import json
from collections import Counter

np.random.seed(42)

df = pd.read_csv("Haseeb.csv")

df.columns = [c.strip().lower() for c in df.columns]

df = df.dropna(subset=["text", "rating"])
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df = df.dropna(subset=["rating"])
df = df[df["rating"].between(1, 5)]
df["rating"] = df["rating"].astype(int)

def clean_text(t):
    t = str(t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^\x00-\x7F]+", " ", t)
    t = re.sub(r"[^a-zA-Z0-9\s\.,!?'\"-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["text"] = df["text"].apply(clean_text)

df = df[df["text"].str.len() >= 10]
df = df[df["text"].str.len() <= 5000]
df = df.drop_duplicates(subset=["text"])
df = df.reset_index(drop=True)

def map_sentiment(r):
    if r <= 2:
        return "Negative"
    elif r == 3:
        return "Neutral"
    else:
        return "Positive"

df["sentiment"] = df["rating"].apply(map_sentiment)

def review_length_bucket(t):
    n = len(t.split())
    if n < 20:
        return 0
    elif n < 60:
        return 1
    else:
        return 2

df["length_label"] = df["text"].apply(review_length_bucket)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

n = len(df)
train_end = int(0.70 * n)
val_end   = int(0.85 * n)

train_df = df.iloc[:train_end].reset_index(drop=True)
val_df   = df.iloc[train_end:val_end].reset_index(drop=True)
test_df  = df.iloc[val_end:].reset_index(drop=True)

os.makedirs("data", exist_ok=True)

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv",   index=False)
test_df.to_csv("data/test.csv",  index=False)

print(f"Total clean samples : {n}")
print(f"Train               : {len(train_df)}")
print(f"Val                 : {len(val_df)}")
print(f"Test                : {len(test_df)}")
print()
print("Sentiment distribution (train):")
print(train_df["sentiment"].value_counts())
print()
print("Length label distribution (train):")
print(train_df["length_label"].value_counts().sort_index())

all_words = []
for text in train_df["text"]:
    all_words.extend(text.lower().split())

freq = Counter(all_words)
vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
for word, count in freq.most_common():
    if count >= 2:
        vocab[word] = len(vocab)

with open("data/vocab.json", "w") as f:
    json.dump(vocab, f)

print(f"\nVocabulary size: {len(vocab)}")
print("Saved: data/train.csv, data/val.csv, data/test.csv, data/vocab.json")

