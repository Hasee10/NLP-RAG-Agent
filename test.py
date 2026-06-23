import torch
import pandas as pd
from pathlib import Path
import os

# Load encoder (should exist)
encoder_path = 'models/encoder_best.pt'
decoder_path = 'models/decoder_best.pt'

if os.path.exists(encoder_path):
    print(f"✓ Loading encoder from {encoder_path}")
    encoder = torch.load(encoder_path, map_location='cpu')
else:
    print(f"✗ Encoder not found at {encoder_path}")
    encoder = None

if os.path.exists(decoder_path):
    print(f"✓ Loading decoder from {decoder_path}")
    decoder = torch.load(decoder_path, map_location='cpu')
else:
    print(f"✗ Decoder not found at {decoder_path} - Still training (Task C in progress)")
    decoder = None

# Load test data
test_contexts_path = 'results/test_contexts.csv'
test_data_path = 'data/test.csv'

if os.path.exists(test_contexts_path):
    print(f"✓ Loading test contexts from {test_contexts_path}")
    test_contexts = pd.read_csv(test_contexts_path)
else:
    print(f"✗ Test contexts not found at {test_contexts_path}")
    test_contexts = None

if os.path.exists(test_data_path):
    print(f"✓ Loading test data from {test_data_path}")
    test_df = pd.read_csv(test_data_path)
else:
    print(f"✗ Test data not found at {test_data_path}")
    test_df = None

print("\n" + "="*80)
print("RAG Agent Testing - Available Components")
print("="*80 + "\n")

# Test on a few samples if data is available
if test_df is not None and test_contexts is not None:
    num_samples = min(5, len(test_df))
    for i in range(num_samples):
        review = test_df.iloc[i]['text']
        sentiment = test_df.iloc[i]['sentiment']
        context = test_contexts.iloc[i]['context'] if 'context' in test_contexts.columns else "No context"
        
        print(f"\n[Sample {i+1}]")
        print(f"Sentiment   : {sentiment}")
        print(f"Review      : {review[:120]}...")
        print(f"RAG Context : {str(context)[:120] if pd.notna(context) else 'No context retrieved'}...")
else:
    print("✗ Cannot test - test data or contexts not available")

print("\n" + "="*80)
print("Status Summary")
print("="*80)
print(f"Encoder Model     : {'✓ Ready' if encoder is not None else '✗ Not found'}")
print(f"Decoder Model     : {'✓ Ready' if decoder is not None else '✗ Still training (wait for Task C to complete)'}")
print(f"Test Contexts     : {'✓ Available' if test_contexts is not None else '✗ Not found'}")
print(f"Test Data         : {'✓ Available' if test_df is not None else '✗ Not found'}")
print("\nNote: Decoder model will be available once Task C training completes.")
print("="*80)