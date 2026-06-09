import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # Use all 3x RTX 3090

import sys
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# ─── Constants ─────────────────────────────────────────────────────────────────
MBTI_TYPES = sorted([
    'INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFP', 'INFP', 'ENFJ',
    'ISFP', 'ISFJ', 'ISTP', 'ISTJ', 'ESTP', 'ESTJ', 'ESFP', 'ESFJ'
])
TYPE_TO_ID = {t: i for i, t in enumerate(MBTI_TYPES)}
ID_TO_TYPE = {i: t for i, t in enumerate(MBTI_TYPES)}

# ─── Dataset ───────────────────────────────────────────────────────────────────
class MBTIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[idx]),
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
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ─── Evaluation ────────────────────────────────────────────────────────────────
def eval_model(model, dataloader, device):
    model.eval()
    predictions, real_values = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # DataParallel: call module directly to get logits safely
            if isinstance(model, torch.nn.DataParallel):
                outputs = model.module(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            real_values.extend(labels.cpu().numpy())

    return accuracy_score(real_values, predictions), predictions, real_values

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("==================================================")
    print("Stage 2: BERT Fine-tuning (3x RTX 3090 DataParallel)")
    print("==================================================")

    csv_path   = '/data/final_preprocessed_mbti.csv'
    best_path  = '/data/best_finetuned_mbti.pt'

    if not os.path.exists(csv_path):
        print(f"ERROR: Preprocessed dataset not found at {csv_path}")
        sys.exit(1)

    # ── Load & label ──────────────────────────────────────────────────────────
    print("Loading preprocessed dataset...")
    df = pd.read_csv(csv_path).dropna(subset=['posts'])
    df['label'] = df['type'].map(TYPE_TO_ID)
    df = df.dropna(subset=['label'])          # drop unknown types
    df['label'] = df['label'].astype(int)

    texts  = df['posts'].tolist()
    labels = df['label'].tolist()
    print(f"Total samples: {len(texts)}")

    # ── Train / Val / Test split (70 / 15 / 15) ───────────────────────────────
    tr_texts, tmp_texts, tr_labels, tmp_labels = train_test_split(
        texts, labels, test_size=0.30, random_state=42, stratify=labels
    )
    va_texts, te_texts, va_labels, te_labels = train_test_split(
        tmp_texts, tmp_labels, test_size=0.50, random_state=42, stratify=tmp_labels
    )
    print(f"Train: {len(tr_texts)} | Val: {len(va_texts)} | Test: {len(te_texts)}")

    # ── Device & GPU check ────────────────────────────────────────────────────
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus     = torch.cuda.device_count()
    print(f"Device: {device}  |  GPUs visible: {n_gpus}")
    for i in range(n_gpus):
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {torch.cuda.get_device_properties(i).name}  ({mem_gb:.1f} GB)")

    # ── Tokenizer & Model ─────────────────────────────────────────────────────
    print("\nLoading bert-base-uncased...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model     = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=16
    )

    if n_gpus > 1:
        print(f"Wrapping with DataParallel across {n_gpus} GPUs...")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    MAX_LEN       = 128
    BATCH_PER_GPU = 64
    TOTAL_BATCH   = BATCH_PER_GPU * max(n_gpus, 1)   # 192 if 3 GPUs
    print(f"Batch size: {TOTAL_BATCH}  ({BATCH_PER_GPU} per GPU)")

    tr_ds = MBTIDataset(tr_texts, tr_labels, tokenizer, MAX_LEN)
    va_ds = MBTIDataset(va_texts, va_labels, tokenizer, MAX_LEN)
    te_ds = MBTIDataset(te_texts, te_labels, tokenizer, MAX_LEN)

    tr_loader = DataLoader(tr_ds, batch_size=TOTAL_BATCH, shuffle=True,  num_workers=4, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=TOTAL_BATCH, shuffle=False, num_workers=4, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=TOTAL_BATCH, shuffle=False, num_workers=4, pin_memory=True)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    EPOCHS       = 4
    WARMUP_STEPS = int(0.1 * len(tr_loader) * EPOCHS)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=len(tr_loader) * EPOCHS
    )

    # ── Training loop (with best-checkpoint saving) ───────────────────────────
    best_val_acc  = 0.0
    best_epoch    = -1
    history       = []

    print(f"\nStarting fine-tuning for {EPOCHS} epochs...")
    overall_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n{'='*50}")
        print(f"  Epoch {epoch}/{EPOCHS}")
        print(f"{'='*50}")

        model.train()
        total_loss = 0

        for batch in tqdm(tr_loader, desc=f"  Training Epoch {epoch}"):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_b       = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_b)

            loss = outputs.loss
            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(tr_loader)

        # ── Validation ────────────────────────────────────────────────────────
        val_acc, _, _ = eval_model(model, va_loader, device)
        epoch_elapsed = time.time() - epoch_start

        print(f"  Train Loss : {avg_loss:.4f}")
        print(f"  Val   Acc  : {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  Elapsed    : {epoch_elapsed:.1f}s")

        history.append({'epoch': epoch, 'loss': avg_loss, 'val_acc': val_acc})

        # ── Best checkpoint ───────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            state_dict   = (model.module.state_dict()
                            if isinstance(model, torch.nn.DataParallel)
                            else model.state_dict())
            torch.save(state_dict, best_path)
            print(f"  ✓ New best model saved → {best_path}  (Val Acc: {val_acc:.4f})")

    total_elapsed = time.time() - overall_start
    print(f"\nTraining complete in {total_elapsed/60:.1f} min")
    print(f"Best Val Acc: {best_val_acc:.4f} at Epoch {best_epoch}")

    print("\n── Training History ─────────────────────────────")
    for h in history:
        marker = " ← BEST" if h['epoch'] == best_epoch else ""
        print(f"  Epoch {h['epoch']}: Loss={h['loss']:.4f}  Val Acc={h['val_acc']*100:.2f}%{marker}")

    # ── Final evaluation on Test set using Best checkpoint ────────────────────
    print(f"\nLoading best checkpoint from {best_path} for final Test evaluation...")
    eval_m = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=16)
    eval_m.load_state_dict(torch.load(best_path, map_location='cpu'))
    eval_m = eval_m.to(device)

    test_acc, test_preds, test_real = eval_model(eval_m, te_loader, device)

    target_names    = [ID_TO_TYPE[i] for i in range(16)]
    present_classes = sorted(set(test_real))
    present_names   = [ID_TO_TYPE[c] for c in present_classes]

    print(f"\n{'='*50}")
    print("  Final Classification Report (Stage 2 — Best BERT)")
    print(f"  Best Checkpoint: Epoch {best_epoch}  |  Test Accuracy: {test_acc*100:.2f}%")
    print(f"{'='*50}")
    print(classification_report(test_real, test_preds, target_names=present_names, digits=4))
    print("==================================================")

if __name__ == '__main__':
    main()
