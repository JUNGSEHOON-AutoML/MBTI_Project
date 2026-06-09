"""
pipeline_finetuned.py
─────────────────────────────────────────────────────────────────────────────
파인튜닝 임베딩 기반 완전 파이프라인

[Step 1] best_finetuned_mbti.pt 로 CLS 임베딩 재추출
         → /data/finetuned_embeddings.npy

[Step 2] Stage 3: 4대 지표별 이진 로지스틱 회귀 재검증
         → /data/finetuned_binary_results.csv
         → /data/finetuned_binary_accuracy_chart.png
         → /data/finetuned_binary_f1_chart.png
         → /data/finetuned_binary_cm_*.png

[Step 3] Stage 5: 16-class SVC (C=1, kernel='rbf', gamma='auto')
         → /data/finetuned_svc_model.joblib
         → /data/finetuned_svc_report.txt
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"   # GPU 0 고장, 1·2 사용

import sys
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ─── Constants ─────────────────────────────────────────────────────────────────
MBTI_TYPES = sorted([
    'INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFP', 'INFP', 'ENFJ',
    'ISFP', 'ISFJ', 'ISTP', 'ISTJ', 'ESTP', 'ESTJ', 'ESFP', 'ESFJ'
])
TYPE_TO_ID = {t: i for i, t in enumerate(MBTI_TYPES)}
ID_TO_TYPE = {i: t for i, t in enumerate(MBTI_TYPES)}

OUT_DIR = '/data'

TRAITS = [
    {'key': 'I_E', 'name': 'I/E', 'pos': 0, 'label0': 'E', 'label1': 'I',  'ref': 96.0},
    {'key': 'N_S', 'name': 'N/S', 'pos': 1, 'label0': 'S', 'label1': 'N',  'ref': 98.0},
    {'key': 'F_T', 'name': 'F/T', 'pos': 2, 'label0': 'T', 'label1': 'F',  'ref': 97.0},
    {'key': 'P_J', 'name': 'P/J', 'pos': 3, 'label0': 'J', 'label1': 'P',  'ref': 94.0},
]

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Fine-tuned BERT → CLS Embedding Extraction
# ══════════════════════════════════════════════════════════════════════════════
def step1_extract_finetuned_embeddings():
    print("\n" + "="*60)
    print("  [Step 1] Fine-tuned BERT CLS Embedding Extraction")
    print("="*60)

    csv_path       = '/data/final_preprocessed_mbti.csv'
    model_path     = '/data/best_finetuned_mbti.pt'
    emb_out        = '/data/finetuned_embeddings.npy'
    label_out      = '/data/finetuned_labels.npy'

    for p in [csv_path, model_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found"); sys.exit(1)

    # Load dataset
    print("  Loading preprocessed dataset...")
    df = pd.read_csv(csv_path).dropna(subset=['posts'])
    texts  = df['posts'].tolist()
    labels = df['type'].tolist()
    print(f"  Total samples : {len(texts)}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"  Device : {device}  |  GPUs : {n_gpus}")

    # Load tokenizer and fine-tuned model
    print("  Loading fine-tuned BertForSequenceClassification ...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=16
    )
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print("  Model loaded successfully.")

    # Tokenize all texts
    MAX_LEN    = 128
    BATCH_SIZE = 512   # inference → larger batch OK

    print("  Tokenizing ...")
    t0 = time.time()
    enc = tokenizer(
        texts,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    print(f"  Tokenization done in {time.time()-t0:.1f}s")

    dataset    = TensorDataset(enc['input_ids'], enc['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Extract CLS embeddings
    print("  Extracting [CLS] embeddings from fine-tuned model ...")
    all_embs = []
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (b_ids, b_mask) in enumerate(dataloader):
            b_ids  = b_ids.to(device)
            b_mask = b_mask.to(device)
            # Use bert encoder directly (ignore classifier head)
            out = model.bert(input_ids=b_ids, attention_mask=b_mask)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_embs.append(cls)
            if (batch_idx + 1) % 20 == 0:
                done = (batch_idx + 1) * BATCH_SIZE
                pct  = min(done / len(texts) * 100, 100)
                print(f"    {min(done, len(texts))}/{len(texts)} ({pct:.1f}%) "
                      f"| {time.time()-t0:.1f}s")

    all_embs = np.concatenate(all_embs, axis=0)
    elapsed  = time.time() - t0
    print(f"  Extracted shape : {all_embs.shape}  | {elapsed:.1f}s")

    np.save(emb_out,   all_embs)
    np.save(label_out, np.array(labels))
    print(f"  Saved → {emb_out}")
    print(f"  Saved → {label_out}")
    return all_embs, np.array(labels)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Stage 3 — Binary Logistic Regression (fine-tuned embeddings)
# ══════════════════════════════════════════════════════════════════════════════
def extract_binary(labels, pos):
    mp = {'I':1,'E':0,'N':1,'S':0,'F':1,'T':0,'P':1,'J':0}
    return np.array([mp.get(str(l)[pos], -1) for l in labels])

def step2_binary_stage3(X, y_labels):
    print("\n" + "="*60)
    print("  [Step 2] Stage 3: Binary Logistic Regression")
    print("  (Fine-tuned BERT embeddings)")
    print("="*60)

    summary_rows = []
    acc_values   = []
    f1_data      = {}

    for trait in TRAITS:
        key, name = trait['key'], trait['name']
        l0, l1    = trait['label0'], trait['label1']
        ref       = trait['ref']

        print(f"\n  ── [{key}] {name} ──")
        y_bin = extract_binary(y_labels, trait['pos'])
        mask  = y_bin >= 0
        Xu, yu = X[mask], y_bin[mask]
        dist = {l0: int((yu==0).sum()), l1: int((yu==1).sum())}
        print(f"  Label dist : {dist}")

        Xtr, Xte, ytr, yte = train_test_split(
            Xu, yu, test_size=0.2, random_state=42, stratify=yu
        )

        t0  = time.time()
        clf = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1, C=1.0)
        clf.fit(Xtr, ytr)
        elapsed = time.time() - t0

        ypred = clf.predict(Xte)
        acc   = accuracy_score(yte, ypred)

        rpt_str  = classification_report(yte, ypred, target_names=[l0, l1], digits=4)
        rpt_dict = classification_report(yte, ypred, target_names=[l0, l1],
                                         digits=4, output_dict=True)

        print(f"  Accuracy : {acc*100:.2f}%   (ref ~{ref:.0f}%)")
        print(f"  Time     : {elapsed:.1f}s")
        for line in rpt_str.splitlines():
            print(f"    {line}")

        # Confusion matrix
        cm      = confusion_matrix(yte, ypred)
        fig, ax = plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay(cm, display_labels=[l0, l1]).plot(
            ax=ax, colorbar=True, cmap='Blues')
        ax.set_title(f'[Finetuned] {key}  Acc={acc*100:.2f}%', fontsize=11)
        cm_path = os.path.join(OUT_DIR, f'finetuned_binary_cm_{key}.png')
        fig.tight_layout(); fig.savefig(cm_path, dpi=150); plt.close(fig)

        f0 = rpt_dict[l0]['f1-score']
        f1 = rpt_dict[l1]['f1-score']
        summary_rows.append({
            'Trait': key, 'Name': name,
            'Accuracy (%)': round(acc*100, 4),
            'Reference (%)': ref,
            'Gap (pp)': round(acc*100 - ref, 2),
            f'F1_{l0}': round(f0, 4),
            f'F1_{l1}': round(f1, 4),
            'Macro F1': round(rpt_dict['macro avg']['f1-score'], 4),
        })
        acc_values.append(acc * 100)
        f1_data[key] = {l0: f0, l1: f1}

    # CSV
    df_csv = pd.DataFrame(summary_rows)
    csv_p  = os.path.join(OUT_DIR, 'finetuned_binary_results.csv')
    df_csv.to_csv(csv_p, index=False, encoding='utf-8-sig')

    # Summary table
    print(f"\n  {'─'*52}")
    print(f"  {'Trait':<8} {'Accuracy':>10}  {'Ref':>6}  {'Gap':>8}")
    print(f"  {'─'*8} {'─'*10}  {'─'*6}  {'─'*8}")
    for r in summary_rows:
        print(f"  {r['Trait']:<8} {r['Accuracy (%)']:>9.2f}%  "
              f"{r['Reference (%)']:>5.0f}%  {r['Gap (pp)']:>+8.2f}")
    print(f"  {'─'*52}")

    # Accuracy chart
    x     = np.arange(len(TRAITS))
    w     = 0.35
    keys  = [t['key'] for t in TRAITS]
    refs  = [t['ref']  for t in TRAITS]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#1E1E2E'); ax.set_facecolor('#1E1E2E')
    b1 = ax.bar(x - w/2, acc_values, w, color='#4CAF50', edgecolor='white', alpha=0.9, label='Finetuned Accuracy')
    b2 = ax.bar(x + w/2, refs,       w, color='#FF7043', edgecolor='white', alpha=0.7, label='Reference Target')
    for bar, v in zip(b1, acc_values):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:.2f}%',
                ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')
    for bar, v in zip(b2, refs):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f'{v:.0f}%',
                ha='center', va='bottom', fontsize=9, color='#FFCC80', fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(keys, color='white', fontsize=11)
    ax.set_ylim(0, 110); ax.set_ylabel('Accuracy (%)', color='white')
    ax.set_title('Stage 3 (Fine-tuned Embeddings): Binary Accuracy by Trait',
                 color='white', fontweight='bold', fontsize=12)
    ax.tick_params(colors='white')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color('#555')
    ax.yaxis.grid(True, color='#333', linestyle='--', alpha=0.7); ax.set_axisbelow(True)
    ax.legend(facecolor='#2E2E3E', edgecolor='#555', labelcolor='white', fontsize=10)
    acc_chart = os.path.join(OUT_DIR, 'finetuned_binary_accuracy_chart.png')
    fig.tight_layout(); fig.savefig(acc_chart, dpi=150, bbox_inches='tight',
                                    facecolor='#1E1E2E'); plt.close(fig)

    # F1 chart
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
    fig.patch.set_facecolor('#1E1E2E')
    fig.suptitle('Stage 3 (Fine-tuned): F1 per Class', color='white',
                 fontsize=13, fontweight='bold')
    palette = ['#EF5350', '#42A5F5']
    for t, ax_s in zip(TRAITS, axes):
        l0, l1 = t['label0'], t['label1']
        fv = [f1_data[t['key']][l0], f1_data[t['key']][l1]]
        bars = ax_s.bar([l0, l1], fv, color=palette, edgecolor='white', width=0.5)
        for bar, v in zip(bars, fv):
            ax_s.text(bar.get_x()+bar.get_width()/2, v+0.01,
                      f'{v:.3f}', ha='center', fontsize=9, color='white', fontweight='bold')
        ax_s.set_facecolor('#1E1E2E'); ax_s.set_title(t['key'], color='white', fontweight='bold')
        ax_s.set_ylim(0, 1.12); ax_s.tick_params(colors='white')
        for sp in ['top','right']: ax_s.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax_s.spines[sp].set_color('#555')
        ax_s.yaxis.grid(True, color='#333', linestyle='--', alpha=0.7); ax_s.set_axisbelow(True)
    f1_chart = os.path.join(OUT_DIR, 'finetuned_binary_f1_chart.png')
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(f1_chart, dpi=150, bbox_inches='tight', facecolor='#1E1E2E'); plt.close(fig)

    print(f"  CSV   → {csv_p}")
    print(f"  Chart → {acc_chart}")
    print(f"  Chart → {f1_chart}")
    return summary_rows


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Stage 5 — 16-class SVC (C=1, rbf, gamma='auto')
# ══════════════════════════════════════════════════════════════════════════════
def step3_svc_stage5(X, y_labels):
    print("\n" + "="*60)
    print("  [Step 3] Stage 5: 16-class SVC  (C=1, rbf, gamma=auto)")
    print("  (Fine-tuned BERT embeddings)")
    print("="*60)

    svc_path = os.path.join(OUT_DIR, 'finetuned_svc_model.joblib')

    # Map string labels → int IDs
    y_ids = np.array([TYPE_TO_ID.get(str(lbl), -1) for lbl in y_labels])
    valid = y_ids >= 0
    X, y_ids = X[valid], y_ids[valid]
    print(f"  Samples after filtering : {len(X)}")

    # 80/20 split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_ids, test_size=0.2, random_state=42, stratify=y_ids
    )
    print(f"  Train : {len(ytr)}  |  Test : {len(yte)}")

    # SVC requires O(N^2) memory — use stratified 20k subset for training
    TRAIN_LIMIT = 20000
    if len(Xtr) > TRAIN_LIMIT:
        print(f"\n  SVC RBF is O(N^2) — subsampling {TRAIN_LIMIT} stratified samples for training")
        Xtr, _, ytr, _ = train_test_split(
            Xtr, ytr, train_size=TRAIN_LIMIT, random_state=42, stratify=ytr
        )
    print(f"  SVC train size : {len(Xtr)}")

    print(f"\n  Training SVC(C=1, kernel='rbf', gamma='auto') ...")
    t0  = time.time()
    clf = SVC(C=1, kernel='rbf', gamma='auto', random_state=42, verbose=False)
    clf.fit(Xtr, ytr)
    elapsed = time.time() - t0
    print(f"  SVC training done in {elapsed:.1f}s")

    # Save
    joblib.dump(clf, svc_path)
    print(f"  Model saved → {svc_path}")

    # Evaluate on full test set
    print("  Evaluating on test set ...")
    t0    = time.time()
    ypred = clf.predict(Xte)
    print(f"  Inference done in {time.time()-t0:.1f}s")

    acc  = accuracy_score(yte, ypred)
    tnames  = [ID_TO_TYPE[i] for i in range(16)]
    present = sorted(set(yte))
    pnames  = [ID_TO_TYPE[c] for c in present]
    rpt_str = classification_report(yte, ypred, labels=present,
                                    target_names=pnames, digits=4)

    print(f"\n  Test Accuracy : {acc*100:.2f}%  (target ~90%)")
    print(f"\n  Classification Report:")
    for line in rpt_str.splitlines():
        print(f"    {line}")

    # Save report
    rpt_path = os.path.join(OUT_DIR, 'finetuned_svc_report.txt')
    with open(rpt_path, 'w') as f:
        f.write(f"Stage 5 SVC (C=1, rbf, gamma=auto) — Fine-tuned Embeddings\n")
        f.write(f"Test Accuracy: {acc*100:.4f}%\n\n")
        f.write(rpt_str)
    print(f"  Report saved → {rpt_path}")

    # Per-class accuracy bar chart
    rpt_dict  = classification_report(yte, ypred, labels=present,
                                      target_names=pnames, digits=4, output_dict=True)
    f1_scores = [rpt_dict[n]['f1-score'] for n in pnames]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1E1E2E'); ax.set_facecolor('#1E1E2E')
    bars = ax.bar(pnames, f1_scores, color='#42A5F5', edgecolor='white', alpha=0.9)
    for bar, v in zip(bars, f1_scores):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                f'{v:.3f}', ha='center', va='bottom', fontsize=7,
                color='white', fontweight='bold')
    ax.axhline(acc, color='#FF7043', linestyle='--', linewidth=1.5,
               label=f'Overall Acc={acc*100:.2f}%')
    ax.set_xlabel('MBTI Type', color='white', fontsize=11)
    ax.set_ylabel('F1 Score', color='white', fontsize=11)
    ax.set_title('Stage 5 SVC (Fine-tuned) — F1 per MBTI Type',
                 color='white', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1); ax.tick_params(colors='white')
    ax.legend(facecolor='#2E2E3E', edgecolor='#555', labelcolor='white')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color('#555')
    ax.yaxis.grid(True, color='#333', linestyle='--', alpha=0.7); ax.set_axisbelow(True)
    svc_chart = os.path.join(OUT_DIR, 'finetuned_svc_f1_chart.png')
    fig.tight_layout(); fig.savefig(svc_chart, dpi=150, bbox_inches='tight',
                                    facecolor='#1E1E2E'); plt.close(fig)
    print(f"  Chart saved → {svc_chart}")
    return acc


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    total_start = time.time()
    print("=" * 60)
    print("  Fine-tuned Embedding Pipeline  (Step 1 → 2 → 3)")
    print("=" * 60)

    # ── Step 1 ────────────────────────────────────────────────────────────────
    emb_path   = '/data/finetuned_embeddings.npy'
    label_path = '/data/finetuned_labels.npy'

    if os.path.exists(emb_path) and os.path.exists(label_path):
        print("\n[Step 1] Found existing finetuned_embeddings.npy — loading...")
        X        = np.load(emb_path)
        y_labels = np.load(label_path)
        print(f"  Loaded: {X.shape}")
    else:
        X, y_labels = step1_extract_finetuned_embeddings()

    # ── Step 2 ────────────────────────────────────────────────────────────────
    step2_binary_stage3(X, y_labels)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    step3_svc_stage5(X, y_labels)

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL STEPS COMPLETE!  Total time: {total/60:.1f} min")
    print(f"{'='*60}")
