import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# Constants
MBTI_TYPES = [
    'INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFP', 'INFP', 'ENFJ',
    'ISFP', 'ISFJ', 'ISTP', 'ISTJ', 'ESTP', 'ESTJ', 'ESFP', 'ESFJ'
]
TYPE_TO_ID = {t: i for i, t in enumerate(MBTI_TYPES)}
ID_TO_TYPE = {i: t for i, t in enumerate(MBTI_TYPES)}

class MBTIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def eval_model(model, dataloader, device):
    model.eval()
    predictions = []
    real_values = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.cpu().numpy())
            
    return accuracy_score(real_values, predictions), predictions, real_values

def main():
    print("==================================================")
    print("Stage 2 (Fine-tuning): BertForSequenceClassification")
    print("==================================================")
    
    csv_path = '/data/final_preprocessed_mbti.csv'
    model_save_path = '/data/bert_mbti_epoch2.pt'
    
    if not os.path.exists(csv_path):
        print(f"ERROR: Preprocessed dataset missing at {csv_path}")
        sys.exit(1)
        
    # Load dataset
    print("Loading preprocessed dataset...")
    df = pd.read_csv(csv_path).dropna(subset=['posts'])
    
    # Map type to label ID
    df['label'] = df['type'].map(TYPE_TO_ID)
    
    texts = df['posts'].tolist()
    labels = df['label'].tolist()
    
    # Split into Train and Validation (80-20)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Train samples: {len(train_texts)}, Validation samples: {len(val_texts)}")
    
    # Setup Device & GPUS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Tokenizer and Model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=16)
    
    # DataParallel wrapper if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Wrapping model with DataParallel on {torch.cuda.device_count()} GPUs...")
        model = torch.nn.DataParallel(model)
        
    model = model.to(device)
    
    # Datasets and Loaders
    max_len = 128
    batch_size = 64 if torch.cuda.device_count() > 1 else 32
    
    train_dataset = MBTIDataset(train_texts, train_labels, tokenizer, max_len)
    val_dataset = MBTIDataset(val_texts, val_labels, tokenizer, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    
    # Check if Epoch 2 weights already exist to bypass training
    if os.path.exists(model_save_path):
        print(f"Fine-tuned model weights found at {model_save_path}. Skipping fine-tuning training loop.")
    else:
        # Training loop for 4 epochs
        print("Starting fine-tuning...")
        epochs = 4
        for epoch in range(1, epochs + 1):
            print(f"\n--- Epoch {epoch}/{epochs} ---")
            model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = outputs.loss
                if isinstance(model, torch.nn.DataParallel):
                    loss = loss.mean()
                    
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            avg_train_loss = total_loss / len(train_loader)
            val_acc, _, _ = eval_model(model, val_loader, device)
            print(f"Epoch {epoch} finished - Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save model weights at Epoch 2
            if epoch == 2:
                print(f"Saving Epoch 2 weights to {model_save_path}...")
                state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state_dict, model_save_path)
            
    # Load Epoch 2 weights to evaluate final predictions
    print(f"\nLoading Epoch 2 weights from {model_save_path} for final evaluation...")
    eval_model_instance = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=16)
    eval_model_instance.load_state_dict(torch.load(model_save_path))
    eval_model_instance = eval_model_instance.to(device)
    
    # Evaluate Epoch 2 model
    print("Evaluating Epoch 2 model on validation set...")
    val_acc, val_preds, val_real = eval_model(eval_model_instance, val_loader, device)
    
    # Print Classification Report
    print(f"\n--- Classification Report (Stage 2 - Fine-tuned BERT Epoch 2) ---")
    print(f"Validation Accuracy: {val_acc:.4f}")
    target_names = [ID_TO_TYPE[i] for i in range(16)]
    # Filter targets based on present classes in val_real to avoid warnings
    present_classes = np.unique(val_real)
    present_names = [ID_TO_TYPE[c] for c in present_classes]
    
    print(classification_report(val_real, val_preds, target_names=present_names))
    print("==================================================")

if __name__ == '__main__':
    main()
