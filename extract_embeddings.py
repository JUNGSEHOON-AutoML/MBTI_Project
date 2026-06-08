import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

def main():
    print("==================================================")
    print("Embedding Extraction using bert-base-uncased")
    print("==================================================")
    
    csv_path = '/data/final_preprocessed_mbti.csv'
    emb_output = '/data/bert_embeddings.npy'
    label_output = '/data/labels.npy'
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Preprocessed dataset missing at {csv_path}")
        sys.exit(1)
        
    # Load dataset
    print("Loading preprocessed dataset...")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    
    # Drop rows with NaN posts (if any)
    df = df.dropna(subset=['posts'])
    texts = df['posts'].tolist()
    labels = df['type'].tolist()
    
    # Initialize Tokenizer and Model
    print("Initializing BERT model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model = model.to(device)
    model.eval()
    
    # Tokenization in batches
    print("Tokenizing texts...")
    max_len = 128
    batch_size = 256
    
    # Tokenize all texts
    encodings = tokenizer(
        texts,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Extract Embeddings
    print("Extracting [CLS] embeddings on GPU...")
    embeddings = []
    
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (b_ids, b_mask) in enumerate(tqdm(dataloader)):
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            
            # Forward pass
            outputs = model(input_ids=b_ids, attention_mask=b_mask)
            
            # [CLS] token is the first token (index 0) of the last hidden state
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_embeddings)
            
    # Concatenate all batches
    all_embeddings = np.concatenate(embeddings, axis=0)
    print(f"Extracted embeddings shape: {all_embeddings.shape}")
    
    # Save results
    print(f"Saving embeddings to {emb_output}...")
    np.save(emb_output, all_embeddings)
    
    print(f"Saving labels to {label_output}...")
    np.save(label_output, np.array(labels))
    
    elapsed = time.time() - start_time
    print(f"Embedding extraction completed in {elapsed:.1f}s!")
    print("==================================================")

if __name__ == '__main__':
    main()
