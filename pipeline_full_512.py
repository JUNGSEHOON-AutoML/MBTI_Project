"""
pipeline_full_512.py
─────────────────────────────────────────────────────────────────────────────
[Step 1] BERT Fine-tuning
         max_length=512, batch=16/GPU, 4 epochs, lr=2e-5
         → /data/best_finetuned_512.pt

[Step 2] Fine-tuned CLS Embedding 재추출 (512 토큰 기준)
         → /data/finetuned_embeddings_512.npy

[Step 3] Stage 5: StandardScaler + SVC(C=1, rbf, gamma='auto')
         → 전체 91k Train 사용 (서브샘플링 없음)
         → /data/final_svc_512_report.txt
─────────────────────────────────────────────────────────────────────────────
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"   # GPU 0 고장

import sys
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import (BertTokenizer, BertForSequenceClassification,
                          AdamW, get_linear_schedule_with_warmup)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# ── 상수 ──────────────────────────────────────────────────────────────────────
MBTI_TYPES = sorted([
    'INFJ','ENTP','INTP','INTJ','ENTJ','ENFP','INFP','ENFJ',
    'ISFP','ISFJ','ISTP','ISTJ','ESTP','ESTJ','ESFP','ESFJ'
])
TYPE_TO_ID = {t: i for i, t in enumerate(MBTI_TYPES)}
ID_TO_TYPE = {i: t for i, t in enumerate(MBTI_TYPES)}

CSV_PATH       = '/data/final_preprocessed_mbti.csv'
BEST_PT        = '/data/best_finetuned_512.pt'
EMB_OUT        = '/data/finetuned_embeddings_512.npy'
LABEL_OUT      = '/data/finetuned_labels_512.npy'
SVC_MODEL_PATH = '/data/final_svc_512_model.joblib'
SVC_REPORT     = '/data/final_svc_512_report.txt'
SVC_CHART      = '/data/final_svc_512_f1_chart.png'

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
MAX_LEN        = 512    # 원본과 동일
BATCH_PER_GPU  = 16     # GPU당 16
EPOCHS         = 4
LR             = 2e-5

# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
class MBTIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
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
            'input_ids':      enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ══════════════════════════════════════════════════════════════════════════════
# 평가 함수
# ══════════════════════════════════════════════════════════════════════════════
def eval_model(model, loader, device):
    model.eval()
    preds, reals = [], []
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            m    = model.module if isinstance(model, torch.nn.DataParallel) else model
            out  = m(input_ids=ids, attention_mask=mask)
            _, p = torch.max(out.logits, dim=1)
            preds.extend(p.cpu().numpy())
            reals.extend(batch['label'].numpy())
    return accuracy_score(reals, preds), preds, reals

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: BERT Fine-tuning (max_length=512)
# ══════════════════════════════════════════════════════════════════════════════
def step1_finetune():
    print("\n" + "="*60)
    print("  [Step 1] BERT Fine-tuning")
    print(f"  max_length={MAX_LEN}  batch/GPU={BATCH_PER_GPU}  lr={LR}  epochs={EPOCHS}")
    print("="*60)

    df = pd.read_csv(CSV_PATH).dropna(subset=['posts'])
    df['label'] = df['type'].map(TYPE_TO_ID)
    df = df.dropna(subset=['label']); df['label'] = df['label'].astype(int)
    texts, labels = df['posts'].tolist(), df['label'].tolist()
    print(f"  Total: {len(texts):,}")

    # 70/15/15 split (val 따로 유지)
    tr_t, tmp_t, tr_l, tmp_l = train_test_split(
        texts, labels, test_size=0.30, random_state=42, stratify=labels)
    va_t, te_t, va_l, te_l = train_test_split(
        tmp_t, tmp_l, test_size=0.50, random_state=42, stratify=tmp_l)
    print(f"  Train: {len(tr_t):,} | Val: {len(va_t):,} | Test: {len(te_t):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    total_batch = BATCH_PER_GPU * max(n_gpus, 1)
    print(f"  Device: {device}  GPUs: {n_gpus}  Total batch: {total_batch}")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model     = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=16)
    if n_gpus > 1:
        print(f"  DataParallel × {n_gpus}")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    tr_ds = MBTIDataset(tr_t, tr_l, tokenizer, MAX_LEN)
    va_ds = MBTIDataset(va_t, va_l, tokenizer, MAX_LEN)
    tr_ld = DataLoader(tr_ds, batch_size=total_batch, shuffle=True,
                       num_workers=4, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=total_batch, shuffle=False,
                       num_workers=4, pin_memory=True)

    warmup = int(0.1 * len(tr_ld) * EPOCHS)
    opt    = AdamW(model.parameters(), lr=LR, correct_bias=False)
    sch    = get_linear_schedule_with_warmup(opt, warmup, len(tr_ld)*EPOCHS)

    best_val, best_epoch = 0.0, -1
    history = []

    for epoch in range(1, EPOCHS+1):
        ep_start = time.time()
        print(f"\n{'='*50}\n  Epoch {epoch}/{EPOCHS}\n{'='*50}")
        model.train(); total_loss = 0

        for batch in tqdm(tr_ld, desc=f"  Train E{epoch}"):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbl  = batch['label'].to(device)
            opt.zero_grad()
            out  = model(input_ids=ids, attention_mask=mask, labels=lbl)
            loss = out.loss
            if isinstance(model, torch.nn.DataParallel): loss = loss.mean()
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step()

        avg_loss = total_loss / len(tr_ld)
        val_acc, _, _ = eval_model(model, va_ld, device)
        elapsed = time.time() - ep_start
        print(f"  Train Loss : {avg_loss:.4f}")
        print(f"  Val Acc    : {val_acc*100:.2f}%")
        print(f"  Elapsed    : {elapsed/60:.1f}min")
        history.append({'epoch': epoch, 'loss': avg_loss, 'val_acc': val_acc})

        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            sd = (model.module.state_dict()
                  if isinstance(model, torch.nn.DataParallel)
                  else model.state_dict())
            torch.save(sd, BEST_PT)
            print(f"  ✓ Best saved → {BEST_PT}  (Val={val_acc*100:.2f}%)")

    print(f"\n  Training History:")
    for h in history:
        m = " ← BEST" if h['epoch']==best_epoch else ""
        print(f"    Epoch {h['epoch']}: Loss={h['loss']:.4f}  Val={h['val_acc']*100:.2f}%{m}")
    print(f"\n  Best Val Acc: {best_val*100:.2f}%  at Epoch {best_epoch}")
    return best_val

# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Fine-tuned 임베딩 추출
# ══════════════════════════════════════════════════════════════════════════════
def step2_extract():
    print("\n" + "="*60)
    print("  [Step 2] Fine-tuned CLS 임베딩 추출 (max_length=512)")
    print("="*60)

    df     = pd.read_csv(CSV_PATH).dropna(subset=['posts'])
    texts  = df['posts'].tolist()
    labels = df['type'].tolist()
    print(f"  샘플: {len(texts):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model     = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=16)
    model.load_state_dict(torch.load(BEST_PT, map_location='cpu'))
    model = model.to(device)
    model.eval()
    print("  Best 모델 로드 완료")

    # 배치 토크나이징
    INFER_BATCH = 64   # 512 토큰 기준 추론 배치
    print(f"  토크나이징 + 추출 (batch={INFER_BATCH})...")
    all_embs = []
    t0 = time.time()

    for start in range(0, len(texts), INFER_BATCH):
        chunk = texts[start:start+INFER_BATCH]
        enc   = tokenizer(chunk, max_length=MAX_LEN, padding='max_length',
                          truncation=True, return_tensors='pt')
        with torch.no_grad():
            out = model.bert(
                input_ids      = enc['input_ids'].to(device),
                attention_mask = enc['attention_mask'].to(device)
            )
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_embs.append(cls)

        done = min(start+INFER_BATCH, len(texts))
        if done % 10000 < INFER_BATCH or done == len(texts):
            pct = done/len(texts)*100
            print(f"    {done:,}/{len(texts):,} ({pct:.1f}%) | {time.time()-t0:.0f}s")

    all_embs = np.concatenate(all_embs, axis=0)
    print(f"  추출 완료: {all_embs.shape}  | {(time.time()-t0)/60:.1f}min")

    np.save(EMB_OUT,   all_embs)
    np.save(LABEL_OUT, np.array(labels))
    print(f"  저장 → {EMB_OUT}")
    return all_embs, np.array(labels)

# ══════════════════════════════════════════════════════════════════════════════
# Step 3: StandardScaler + 전체 91k SVC
# ══════════════════════════════════════════════════════════════════════════════
def step3_svc(X, y_labels):
    print("\n" + "="*60)
    print("  [Step 3] Stage 5: Full 91k SVC")
    print("  StandardScaler + SVC(C=1, rbf, gamma='auto')")
    print("="*60)

    y = np.array([TYPE_TO_ID.get(str(l), -1) for l in y_labels])
    valid = y >= 0; X, y = X[valid], y[valid]
    print(f"  유효 샘플: {len(X):,}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Train: {len(y_tr):,} | Test: {len(y_te):,}")

    print("\n  StandardScaler 적용...")
    t0     = time.time()
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    print(f"  스케일링 완료: {time.time()-t0:.1f}s")

    print(f"\n  SVC 학습 (전체 {len(X_tr_s):,}개, 서브샘플링 없음)...")
    print(f"  시작: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t0  = time.time()
    clf = SVC(C=1, kernel='rbf', gamma='auto', random_state=42,
              verbose=True, cache_size=8000)
    clf.fit(X_tr_s, y_tr)
    train_t = time.time() - t0
    print(f"  학습 완료: {train_t/60:.1f}min | Support vectors: {clf.n_support_.sum():,}")

    from sklearn.pipeline import Pipeline
    pipe = Pipeline([('scaler', scaler), ('svc', clf)])
    joblib.dump(pipe, SVC_MODEL_PATH)
    print(f"  모델 저장 → {SVC_MODEL_PATH}")

    print("\n  추론 중...")
    t0    = time.time()
    y_pred= clf.predict(X_te_s)
    inf_t = time.time() - t0
    print(f"  추론 완료: {inf_t:.1f}s")

    acc    = accuracy_score(y_te, y_pred)
    present= sorted(set(y_te))
    pnames = [ID_TO_TYPE[c] for c in present]
    rpt_s  = classification_report(y_te, y_pred, labels=present,
                                   target_names=pnames, digits=4)
    rpt_d  = classification_report(y_te, y_pred, labels=present,
                                   target_names=pnames, digits=4, output_dict=True)

    print(f"\n{'='*60}")
    print(f"  최종 Test Accuracy: {acc*100:.2f}%")
    print(f"{'='*60}")
    for line in rpt_s.splitlines():
        print(f"    {line}")

    # 리포트 저장
    with open(SVC_REPORT, 'w') as f:
        f.write(f"Final SVC — 파인튜닝 임베딩 512토큰 + StandardScaler (전체 91k)\n")
        f.write(f"C=1, kernel='rbf', gamma='auto'\n")
        f.write(f"Train: {len(y_tr):,} | Test: {len(y_te):,}\n")
        f.write(f"Train time: {train_t:.0f}s ({train_t/60:.1f}min)\n")
        f.write(f"Test Accuracy: {acc*100:.4f}%\n\n")
        f.write(rpt_s)
    print(f"\n  리포트 → {SVC_REPORT}")

    # F1 차트
    f1s = [rpt_d[n]['f1-score'] for n in pnames]
    fig, ax = plt.subplots(figsize=(14,5))
    fig.patch.set_facecolor('#1E1E2E'); ax.set_facecolor('#1E1E2E')
    bars = ax.bar(pnames, f1s, color='#66BB6A', edgecolor='white', alpha=0.9)
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                f'{v:.3f}', ha='center', fontsize=7, color='white', fontweight='bold')
    ax.axhline(acc, color='#FF7043', linestyle='--', linewidth=1.5,
               label=f'Acc={acc*100:.2f}%')
    ax.set_xlabel('MBTI Type', color='white'); ax.set_ylabel('F1', color='white')
    ax.set_title('Final SVC (Fine-tuned 512-token + StandardScaler, Full 91k)',
                 color='white', fontweight='bold')
    ax.set_ylim(0,1.1); ax.tick_params(colors='white')
    ax.legend(facecolor='#2E2E3E', edgecolor='#555', labelcolor='white')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color('#555')
    ax.yaxis.grid(True,color='#333',linestyle='--',alpha=0.7); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(SVC_CHART, dpi=150, bbox_inches='tight', facecolor='#1E1E2E')
    plt.close(fig)
    print(f"  차트 → {SVC_CHART}")
    return acc

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    total_start = time.time()
    print("="*60)
    print("  Full Pipeline (512-token)  Step 1→2→3")
    print(f"  시작: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Step 1: Fine-tuning
    if os.path.exists(BEST_PT):
        print(f"\n[Step 1] {BEST_PT} 이미 존재 — 학습 건너뜀")
    else:
        step1_finetune()

    # Step 2: 임베딩 추출
    if os.path.exists(EMB_OUT) and os.path.exists(LABEL_OUT):
        print(f"\n[Step 2] {EMB_OUT} 이미 존재 — 로드")
        X        = np.load(EMB_OUT)
        y_labels = np.load(LABEL_OUT)
        print(f"  Loaded: {X.shape}")
    else:
        X, y_labels = step2_extract()

    # Step 3: SVC
    step3_svc(X, y_labels)

    total = (time.time()-total_start)/60
    print(f"\n{'='*60}")
    print(f"  전체 파이프라인 완료! 총 소요: {total:.1f}분")
    print(f"  종료: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
