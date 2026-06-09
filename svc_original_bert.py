"""
svc_original_bert.py
────────────────────────────────────────────────────────────
원본(학습 안 된) BERT CLS 임베딩 → SVC (C=1, rbf, gamma='auto')
Stage 1에서 추출해둔 /data/bert_embeddings.npy 직접 사용
────────────────────────────────────────────────────────────
"""

import os, sys, time
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# ── 경로 ──────────────────────────────────────────────────────────────────────
EMB_PATH   = '/data/bert_embeddings.npy'
LABEL_PATH = '/data/labels.npy'
MODEL_PATH = '/data/original_bert_svc_model.joblib'
CHART_PATH = '/data/original_bert_svc_f1_chart.png'
REPORT_PATH= '/data/original_bert_svc_report.txt'

# MBTI 정렬 순서
MBTI_TYPES = sorted([
    'INFJ','ENTP','INTP','INTJ','ENTJ','ENFP','INFP','ENFJ',
    'ISFP','ISFJ','ISTP','ISTJ','ESTP','ESTJ','ESFP','ESFJ'
])
ID_TO_TYPE = {i: t for i, t in enumerate(MBTI_TYPES)}
TYPE_TO_ID = {t: i for i, t in enumerate(MBTI_TYPES)}

def main():
    print("=" * 60)
    print("  Final SVC — 원본 BERT CLS 임베딩 (bert_embeddings.npy)")
    print("  Params: C=1, kernel='rbf', gamma='auto'")
    print("=" * 60)

    # ── 로드 ──────────────────────────────────────────────────────────────────
    for p in [EMB_PATH, LABEL_PATH]:
        if not os.path.exists(p):
            print(f"ERROR: {p} 없음"); sys.exit(1)

    print("\n원본 BERT 임베딩 로드 중...")
    X        = np.load(EMB_PATH)
    y_labels = np.load(LABEL_PATH)
    print(f"  Embeddings : {X.shape}")
    print(f"  Labels     : {y_labels.shape} (unique={len(np.unique(y_labels))})")

    # 레이블이 string이면 int로 변환
    if y_labels.dtype.kind in ('U', 'S', 'O'):
        y = np.array([TYPE_TO_ID.get(str(l), -1) for l in y_labels])
    else:
        y = y_labels.astype(int)
    valid = y >= 0
    X, y = X[valid], y[valid]
    print(f"  유효 샘플 : {len(X)}")

    # ── 80/20 분할 ─────────────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(y_tr):,}  |  Test: {len(y_te):,}")

    # ── SVC 학습: 20k 층화 서브샘플 ──────────────────────────────────────────
    TRAIN_LIMIT = 20000
    if len(X_tr) > TRAIN_LIMIT:
        print(f"\n  SVC RBF 는 O(N^2) — {TRAIN_LIMIT:,} 층화 서브샘플 사용")
        X_tr_svc, _, y_tr_svc, _ = train_test_split(
            X_tr, y_tr, train_size=TRAIN_LIMIT, random_state=42, stratify=y_tr
        )
    else:
        X_tr_svc, y_tr_svc = X_tr, y_tr

    print(f"  SVC 학습 크기: {len(X_tr_svc):,}")
    print(f"\n  SVC 학습 시작 (C=1, rbf, gamma='auto') ...")

    t0  = time.time()
    clf = SVC(C=1, kernel='rbf', gamma='auto', random_state=42, verbose=False)
    clf.fit(X_tr_svc, y_tr_svc)
    train_elapsed = time.time() - t0
    print(f"  학습 완료: {train_elapsed:.1f}s")

    # 저장
    joblib.dump(clf, MODEL_PATH)
    print(f"  모델 저장 → {MODEL_PATH}")

    # ── 추론 및 평가 ──────────────────────────────────────────────────────────
    print("\n  테스트셋 추론 중 ...")
    t0    = time.time()
    y_pred = clf.predict(X_te)
    infer_elapsed = time.time() - t0
    print(f"  추론 완료: {infer_elapsed:.1f}s")

    acc     = accuracy_score(y_te, y_pred)
    present = sorted(set(y_te))
    pnames  = [ID_TO_TYPE[c] for c in present]

    rpt_str  = classification_report(y_te, y_pred, labels=present,
                                     target_names=pnames, digits=4)
    rpt_dict = classification_report(y_te, y_pred, labels=present,
                                     target_names=pnames, digits=4, output_dict=True)

    # ── 출력 ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [최종 결과]  Test Accuracy: {acc*100:.2f}%  (목표 ~90%)")
    print(f"{'='*60}")
    print(f"\n  Classification Report:")
    for line in rpt_str.splitlines():
        print(f"    {line}")
    print(f"{'='*60}")

    # ── 텍스트 리포트 저장 ────────────────────────────────────────────────────
    with open(REPORT_PATH, 'w') as f:
        f.write(f"Final SVC — 원본 BERT CLS 임베딩\n")
        f.write(f"C=1, kernel='rbf', gamma='auto'\n")
        f.write(f"Train subset: {len(X_tr_svc):,}  |  Test: {len(y_te):,}\n")
        f.write(f"Train time: {train_elapsed:.1f}s  |  Infer time: {infer_elapsed:.1f}s\n")
        f.write(f"Test Accuracy: {acc*100:.4f}%\n\n")
        f.write(rpt_str)
    print(f"  리포트 저장 → {REPORT_PATH}")

    # ── F1 차트 ───────────────────────────────────────────────────────────────
    f1_scores = [rpt_dict[n]['f1-score'] for n in pnames]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1E1E2E'); ax.set_facecolor('#1E1E2E')
    bars = ax.bar(pnames, f1_scores, color='#26C6DA', edgecolor='white', alpha=0.9)
    for bar, v in zip(bars, f1_scores):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=7, color='white', fontweight='bold')
    ax.axhline(acc, color='#FF7043', linestyle='--', linewidth=1.5,
               label=f'Overall Acc = {acc*100:.2f}%')
    ax.set_xlabel('MBTI Type', color='white', fontsize=11)
    ax.set_ylabel('F1 Score', color='white', fontsize=11)
    ax.set_title('Final SVC (원본 BERT 임베딩) — F1 per MBTI Type',
                 color='white', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1); ax.tick_params(colors='white')
    ax.legend(facecolor='#2E2E3E', edgecolor='#555', labelcolor='white', fontsize=10)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color('#555')
    ax.yaxis.grid(True, color='#333', linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150, bbox_inches='tight', facecolor='#1E1E2E')
    plt.close(fig)
    print(f"  차트 저장  → {CHART_PATH}")
    print(f"\n{'='*60}")
    print(f"  완료!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
