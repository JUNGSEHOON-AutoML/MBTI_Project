"""
svc_full_scaled.py
─────────────────────────────────────────────────────────────────────────────
원본 BERT CLS 임베딩 (bert_embeddings.npy)
→ StandardScaler → SVC(C=1, rbf, gamma='auto')
→ 전체 91k Train 데이터 사용 (서브샘플링 없음)
─────────────────────────────────────────────────────────────────────────────
"""

import os, sys, time
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# ── 경로 ──────────────────────────────────────────────────────────────────────
EMB_PATH    = '/data/bert_embeddings.npy'
LABEL_PATH  = '/data/labels.npy'
MODEL_PATH  = '/data/full_scaled_svc_model.joblib'
CHART_PATH  = '/data/full_scaled_svc_f1_chart.png'
REPORT_PATH = '/data/full_scaled_svc_report.txt'

MBTI_TYPES = sorted([
    'INFJ','ENTP','INTP','INTJ','ENTJ','ENFP','INFP','ENFJ',
    'ISFP','ISFJ','ISTP','ISTJ','ESTP','ESTJ','ESFP','ESFJ'
])
ID_TO_TYPE = {i: t for i, t in enumerate(MBTI_TYPES)}
TYPE_TO_ID = {t: i for i, t in enumerate(MBTI_TYPES)}

def main():
    print("=" * 60)
    print("  Final SVC — 원본 BERT 임베딩 + StandardScaler")
    print("  전체 91k Train 사용 (서브샘플링 없음)")
    print("  C=1, kernel='rbf', gamma='auto'")
    print("=" * 60)

    # ── 로드 ──────────────────────────────────────────────────────────────────
    for p in [EMB_PATH, LABEL_PATH]:
        if not os.path.exists(p):
            print(f"ERROR: {p} 없음"); sys.exit(1)

    print("\n[1] 원본 BERT 임베딩 로드...")
    X        = np.load(EMB_PATH)
    y_labels = np.load(LABEL_PATH)
    print(f"  Embeddings : {X.shape}")
    print(f"  Labels     : {y_labels.shape} (unique={len(np.unique(y_labels))})")

    # 레이블 string → int
    if y_labels.dtype.kind in ('U', 'S', 'O'):
        y = np.array([TYPE_TO_ID.get(str(l), -1) for l in y_labels])
    else:
        y = y_labels.astype(int)
    valid = y >= 0
    X, y = X[valid], y[valid]
    print(f"  유효 샘플 : {len(X):,}")

    # ── 80/20 분할 ─────────────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[2] 분할 완료")
    print(f"  Train : {len(y_tr):,}")
    print(f"  Test  : {len(y_te):,}")

    # ── StandardScaler ────────────────────────────────────────────────────────
    print("\n[3] StandardScaler fit_transform (Train) ...")
    t0     = time.time()
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    print(f"  스케일링 완료: {time.time()-t0:.1f}s")
    print(f"  Train mean≈0: {X_tr_s.mean():.4f}  std≈1: {X_tr_s.std():.4f}")

    # ── SVC 학습 (전체 91k, 서브샘플링 없음) ─────────────────────────────────
    print(f"\n[4] SVC 학습 시작 — 전체 {len(X_tr_s):,}개 샘플")
    print("  (대용량 학습 — 수 시간 소요 가능)")
    print(f"  시작 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    t0  = time.time()
    clf = SVC(C=1, kernel='rbf', gamma='auto', random_state=42,
              verbose=True, cache_size=8000)   # 8GB cache
    clf.fit(X_tr_s, y_tr)
    train_elapsed = time.time() - t0
    print(f"\n  학습 완료: {train_elapsed/60:.1f}분 ({train_elapsed:.0f}s)")
    print(f"  Support vectors: {clf.n_support_.sum():,}")

    # 파이프라인(Scaler+SVC)로 저장
    pipe = Pipeline([('scaler', scaler), ('svc', clf)])
    joblib.dump(pipe, MODEL_PATH)
    print(f"  모델 저장 → {MODEL_PATH}")

    # ── 추론 ──────────────────────────────────────────────────────────────────
    print("\n[5] 테스트셋 추론 중...")
    t0     = time.time()
    y_pred = clf.predict(X_te_s)
    infer_elapsed = time.time() - t0
    print(f"  추론 완료: {infer_elapsed:.1f}s")

    # ── 평가 ──────────────────────────────────────────────────────────────────
    acc     = accuracy_score(y_te, y_pred)
    present = sorted(set(y_te))
    pnames  = [ID_TO_TYPE[c] for c in present]

    rpt_str  = classification_report(y_te, y_pred, labels=present,
                                     target_names=pnames, digits=4)
    rpt_dict = classification_report(y_te, y_pred, labels=present,
                                     target_names=pnames, digits=4, output_dict=True)

    print(f"\n{'='*60}")
    print(f"  [최종 결과]  Test Accuracy: {acc*100:.2f}%")
    print(f"{'='*60}")
    for line in rpt_str.splitlines():
        print(f"    {line}")
    print(f"{'='*60}")

    # ── 리포트 저장 ───────────────────────────────────────────────────────────
    with open(REPORT_PATH, 'w') as f:
        f.write("Final SVC — 원본 BERT 임베딩 + StandardScaler (전체 91k)\n")
        f.write(f"C=1, kernel='rbf', gamma='auto'\n")
        f.write(f"Train: {len(y_tr):,} (전체)  | Test: {len(y_te):,}\n")
        f.write(f"Train time: {train_elapsed:.0f}s ({train_elapsed/60:.1f}min)\n")
        f.write(f"Infer time: {infer_elapsed:.1f}s\n")
        f.write(f"Test Accuracy: {acc*100:.4f}%\n\n")
        f.write(rpt_str)
    print(f"  리포트 → {REPORT_PATH}")

    # ── F1 차트 ───────────────────────────────────────────────────────────────
    f1_scores = [rpt_dict[n]['f1-score'] for n in pnames]
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1E1E2E'); ax.set_facecolor('#1E1E2E')
    bars = ax.bar(pnames, f1_scores, color='#AB47BC', edgecolor='white', alpha=0.9)
    for bar, v in zip(bars, f1_scores):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=7, color='white', fontweight='bold')
    ax.axhline(acc, color='#FF7043', linestyle='--', linewidth=1.5,
               label=f'Accuracy={acc*100:.2f}%')
    ax.set_xlabel('MBTI Type', fontsize=11, color='white')
    ax.set_ylabel('F1 Score', fontsize=11, color='white')
    ax.set_title('Final SVC (Original BERT + StandardScaler, Full 91k) — F1 per MBTI Type',
                 fontsize=12, fontweight='bold', color='white')
    ax.set_ylim(0, 1.1); ax.tick_params(colors='white')
    ax.legend(facecolor='#2E2E3E', edgecolor='#555', labelcolor='white', fontsize=10)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color('#555')
    ax.yaxis.grid(True, color='#333', linestyle='--', alpha=0.7); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150, bbox_inches='tight', facecolor='#1E1E2E')
    plt.close(fig)
    print(f"  차트  → {CHART_PATH}")
    print(f"\n  완료! 총 소요: {(train_elapsed+infer_elapsed)/60:.1f}분")

if __name__ == '__main__':
    main()
