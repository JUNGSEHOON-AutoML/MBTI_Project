"""
Stage 3: MBTI 4대 지표별 이진 분류 (Binary Logistic Regression)
  - Input  : /data/bert_embeddings.npy  (114741 x 768 BERT CLS)
  - Labels : /data/labels.npy           (114741, MBTI 16-class strings)
  - Output :
      /data/binary_results.csv           (지표별 Accuracy / F1 수치)
      /data/binary_report_<TRAIT>.txt    (각 지표 classification_report 전문)
      /data/binary_accuracy_chart.png    (4개 지표 정확도 막대 그래프)
      /data/binary_f1_chart.png          (4개 지표 클래스별 F1 막대 그래프)
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ─── Output Paths ──────────────────────────────────────────────────────────────
OUT_DIR         = '/data'
CSV_PATH        = os.path.join(OUT_DIR, 'binary_results.csv')
ACC_CHART_PATH  = os.path.join(OUT_DIR, 'binary_accuracy_chart.png')
F1_CHART_PATH   = os.path.join(OUT_DIR, 'binary_f1_chart.png')
CM_DIR          = OUT_DIR   # confusion matrices saved here

# ─── Trait Definitions ─────────────────────────────────────────────────────────
# MBTI 4자리: [0]=I/E, [1]=N/S, [2]=F/T, [3]=P/J
# Binary encoding: 1 = 앞 글자(I, N, F, P), 0 = 뒷 글자(E, S, T, J)
TRAITS = [
    {
        'key':    'I_E',
        'name':   'I/E (Introversion vs Extraversion)',
        'pos':    0,
        'label0': 'E',   # binary 0
        'label1': 'I',   # binary 1
        'target': (96.0, 'reference ~96%')
    },
    {
        'key':    'N_S',
        'name':   'N/S (Intuition vs Sensing)',
        'pos':    1,
        'label0': 'S',
        'label1': 'N',
        'target': (98.0, 'reference ~98%')
    },
    {
        'key':    'F_T',
        'name':   'F/T (Feeling vs Thinking)',
        'pos':    2,
        'label0': 'T',
        'label1': 'F',
        'target': (97.0, 'reference ~97%')
    },
    {
        'key':    'P_J',
        'name':   'P/J (Perceiving vs Judging)',
        'pos':    3,
        'label0': 'J',
        'label1': 'P',
        'target': (94.0, 'reference ~94%')
    },
]

# ─── Helper ────────────────────────────────────────────────────────────────────
def extract_binary(labels: np.ndarray, pos: int) -> np.ndarray:
    """Return binary array: 1 if MBTI[pos] is the first character (I/N/F/P), 0 otherwise."""
    first_chars = {'I': 1, 'E': 0,   # pos 0
                   'N': 1, 'S': 0,   # pos 1
                   'F': 1, 'T': 0,   # pos 2
                   'P': 1, 'J': 0}   # pos 3
    return np.array([first_chars.get(str(lbl)[pos], -1) for lbl in labels])

# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Stage 3: Binary Logistic Regression (4 MBTI Traits)")
    print("=" * 60)

    emb_path   = '/data/bert_embeddings.npy'
    label_path = '/data/labels.npy'

    if not os.path.exists(emb_path) or not os.path.exists(label_path):
        print("ERROR: bert_embeddings.npy or labels.npy not found in /data")
        sys.exit(1)

    print("Loading BERT CLS embeddings and MBTI labels...")
    X        = np.load(emb_path)
    y_labels = np.load(label_path)
    print(f"  Embeddings : {X.shape}")
    print(f"  Labels     : {y_labels.shape}  (unique: {len(np.unique(y_labels))})")

    # ── Result storage ────────────────────────────────────────────────────────
    summary_rows = []
    acc_values   = []
    f1_data      = {}   # {trait_key: {label0: f1, label1: f1}}

    # ══════════════════════════════════════════════════════════════════════════
    for trait in TRAITS:
        key    = trait['key']
        name   = trait['name']
        pos    = trait['pos']
        l0     = trait['label0']
        l1     = trait['label1']
        target = trait['target'][0]

        print(f"\n{'─'*60}")
        print(f"  [{key}]  {name}")
        print(f"{'─'*60}")

        # Extract binary target
        y_bin = extract_binary(y_labels, pos)
        valid_mask = y_bin >= 0
        X_use = X[valid_mask]
        y_use = y_bin[valid_mask]

        dist = {l0: int((y_use == 0).sum()), l1: int((y_use == 1).sum())}
        print(f"  Label dist : {dist}")

        # Train / Test split 80:20
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_use, y_use, test_size=0.2, random_state=42, stratify=y_use
        )
        print(f"  Train: {len(y_tr)}  |  Test: {len(y_te)}")

        # Logistic Regression
        t0  = time.time()
        clf = LogisticRegression(max_iter=2000, random_state=42, n_jobs=-1, C=1.0)
        clf.fit(X_tr, y_tr)
        elapsed = time.time() - t0

        y_pred = clf.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)

        # classification_report
        target_names = [l0, l1]
        report_str   = classification_report(y_te, y_pred, target_names=target_names, digits=4)
        report_dict  = classification_report(y_te, y_pred, target_names=target_names,
                                              digits=4, output_dict=True)

        # Print
        print(f"\n  Accuracy : {acc*100:.2f}%   (reference target ≈ {target:.0f}%)")
        print(f"  Train time: {elapsed:.1f}s")
        print(f"\n  Classification Report:")
        for line in report_str.splitlines():
            print(f"    {line}")

        # Save individual text report
        rpt_path = os.path.join(OUT_DIR, f'binary_report_{key}.txt')
        with open(rpt_path, 'w') as f:
            f.write(f"[{key}] {name}\n")
            f.write(f"Accuracy: {acc*100:.4f}%\n")
            f.write(f"Train time: {elapsed:.1f}s\n\n")
            f.write(report_str)
        print(f"  → Report saved: {rpt_path}")

        # Confusion Matrix
        cm        = confusion_matrix(y_te, y_pred)
        fig, ax   = plt.subplots(figsize=(5, 4))
        disp      = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(ax=ax, colorbar=True, cmap='Blues')
        ax.set_title(f'Confusion Matrix — {key}\nAccuracy: {acc*100:.2f}%', fontsize=11)
        cm_path   = os.path.join(CM_DIR, f'binary_cm_{key}.png')
        fig.tight_layout()
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"  → Confusion matrix: {cm_path}")

        # Collect summary
        f0 = report_dict[l0]['f1-score']
        f1 = report_dict[l1]['f1-score']
        summary_rows.append({
            'Trait':         key,
            'Name':          name,
            'Accuracy (%)':  round(acc * 100, 4),
            'Target (%)':    target,
            'Gap (pp)':      round(acc * 100 - target, 2),
            f'F1_{l0}':      round(f0, 4),
            f'F1_{l1}':      round(f1, 4),
            'Macro F1':      round(report_dict['macro avg']['f1-score'], 4),
            'Train_time(s)': round(elapsed, 1),
        })
        acc_values.append(acc * 100)
        f1_data[key] = {l0: f0, l1: f1}

    # ══════════════════════════════════════════════════════════════════════════
    # Save CSV Summary
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\n{'='*60}")
    print(f"  CSV summary saved → {CSV_PATH}")

    # Print summary table
    print(f"\n{'━'*60}")
    print(f"  {'Trait':<8} {'Accuracy':>10}  {'Target':>8}  {'Gap(pp)':>8}")
    print(f"  {'─'*8} {'─'*10}  {'─'*8}  {'─'*8}")
    for row in summary_rows:
        gap_str = f"{row['Gap (pp)']:+.2f}"
        print(f"  {row['Trait']:<8} {row['Accuracy (%)']:>9.2f}%  "
              f"{row['Target (%)']:>7.0f}%  {gap_str:>8}")
    print(f"{'━'*60}")

    # ── Chart 1: Accuracy Bar Chart ───────────────────────────────────────────
    trait_keys    = [t['key']    for t in TRAITS]
    target_accs   = [t['target'][0] for t in TRAITS]
    colors_actual = ['#4C9BE8', '#4C9BE8', '#4C9BE8', '#4C9BE8']
    colors_target = ['#FF7043', '#FF7043', '#FF7043', '#FF7043']

    x      = np.arange(len(trait_keys))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#1E1E2E')
    ax.set_facecolor('#1E1E2E')

    bars1 = ax.bar(x - width/2, acc_values,   width, label='Actual Accuracy',
                   color='#4C9BE8', edgecolor='white', linewidth=0.5, alpha=0.9)
    bars2 = ax.bar(x + width/2, target_accs,  width, label='Reference Target',
                   color='#FF7043', edgecolor='white', linewidth=0.5, alpha=0.7)

    # Value labels on bars
    for bar, val in zip(bars1, acc_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9,
                color='white', fontweight='bold')
    for bar, val in zip(bars2, target_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9,
                color='#FFCC80', fontweight='bold')

    ax.set_xlabel('MBTI Trait Dimension', color='white', fontsize=11)
    ax.set_ylabel('Accuracy (%)', color='white', fontsize=11)
    ax.set_title('Stage 3: Binary Logistic Regression — Accuracy by Trait\n'
                 '(BERT CLS Embeddings, 80/20 Train-Test Split)',
                 color='white', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(trait_keys, color='white', fontsize=11)
    ax.set_ylim(0, 110)
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('#555')
    ax.spines['left'].set_color('#555')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='#333', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    patch1 = mpatches.Patch(color='#4C9BE8', label='Actual Accuracy')
    patch2 = mpatches.Patch(color='#FF7043', label='Reference Target')
    ax.legend(handles=[patch1, patch2], facecolor='#2E2E3E', edgecolor='#555',
              labelcolor='white', fontsize=10)

    fig.tight_layout()
    fig.savefig(ACC_CHART_PATH, dpi=150, bbox_inches='tight', facecolor='#1E1E2E')
    plt.close(fig)
    print(f"  Accuracy chart saved → {ACC_CHART_PATH}")

    # ── Chart 2: F1 Score by Class ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
    fig.patch.set_facecolor('#1E1E2E')
    fig.suptitle('Stage 3: Binary Classification — F1 Score per Class',
                 color='white', fontsize=13, fontweight='bold')

    palette = ['#EF5350', '#42A5F5']
    for ax_i, (trait, ax_sub) in enumerate(zip(TRAITS, axes)):
        key = trait['key']
        l0  = trait['label0']
        l1  = trait['label1']
        f_vals = [f1_data[key][l0], f1_data[key][l1]]

        bars = ax_sub.bar([l0, l1], f_vals, color=palette, edgecolor='white',
                          linewidth=0.5, width=0.5)
        for bar, val in zip(bars, f_vals):
            ax_sub.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=9, color='white', fontweight='bold')

        ax_sub.set_facecolor('#1E1E2E')
        ax_sub.set_title(key, color='white', fontsize=11, fontweight='bold')
        ax_sub.set_xlabel('Class', color='white')
        if ax_i == 0:
            ax_sub.set_ylabel('F1 Score', color='white')
        ax_sub.set_ylim(0, 1.12)
        ax_sub.tick_params(colors='white')
        for sp in ['top', 'right']:
            ax_sub.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']:
            ax_sub.spines[sp].set_color('#555')
        ax_sub.yaxis.grid(True, color='#333', linestyle='--', alpha=0.7)
        ax_sub.set_axisbelow(True)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(F1_CHART_PATH, dpi=150, bbox_inches='tight', facecolor='#1E1E2E')
    plt.close(fig)
    print(f"  F1 chart saved     → {F1_CHART_PATH}")
    print(f"{'='*60}")
    print("  Stage 3 COMPLETE!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
