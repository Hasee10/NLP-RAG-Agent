import gzip
import json
import pandas as pd
import random

# ----------------------------
# CONFIG (UPDATED FOR 5-CORE)
# ----------------------------

DATASETS = {
    "TZ.csv": {
        "files": [
            "Industrial_and_Scientific.json.gz",
            "Digital_Music.json.gz",
            "Musical_Instruments.json.gz",
            "Prime_Pantry.json.gz"
        ],
        "size": 42000
    }
}

# ----------------------------
# READ FUNCTION
# ----------------------------

def read_json_gz(file_path):
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                review = json.loads(line)
                if 'reviewText' in review and 'overall' in review:
                    data.append({
                        "text": review['reviewText'],
                        "rating": review['overall']
                    })
            except:
                continue
    return data

# ----------------------------
# MAIN
# ----------------------------

for output_file, config in DATASETS.items():
    print(f"\nProcessing {output_file}...")

    all_data = []

    for file in config["files"]:
        print(f"Reading {file}...")
        data = read_json_gz(file)
        all_data.extend(data)

    print(f"Collected: {len(all_data)}")

    # Shuffle
    random.shuffle(all_data)

    # Sample required size
    final_data = all_data[:config["size"]]

    df = pd.DataFrame(final_data)

    # Save
    df.to_csv(output_file, index=False)

    print(f"Saved {output_file} with {len(df)} rows")

print("\nDONE ✅")