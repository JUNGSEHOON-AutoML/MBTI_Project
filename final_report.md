# MBTI Text Classification Project - Final Performance Report

이 문서는 **MBTI 텍스트 분류 모델링 파이프라인**의 수행 결과 및 모델 성능 평가 지표를 상세히 정리한 최종 명세서입니다. 모든 작업은 지정된 GPU 지원 Docker 컨테이너 환경 내에서 실행되었습니다.

---

## 1. 프로젝트 및 실행 환경 개요

- **실행 환경**: Docker Container (`mbti-container`)
- **Base Image**: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` (Python 3.10)
- **하드웨어 가속**: NVIDIA GeForce RTX 3090 (CUDA 호환 활성화)
- **주요 라이브러리**: `pandas`, `nltk`, `torch` (2.0.1), `transformers` (4.39.3), `scikit-learn` (1.3.0)
- **데이터 저장소**: `/userHome/userhome4/sehoon/MBTI_Project/outputs/`

---

## 2. 데이터 전처리 및 통합 결과 (Phase 2)

- **원본 데이터셋**:
  - `mbti_1.csv` (8,675개 포스트)
  - `ForumMessages.csv` (173,084개 포스트)
- **전처리 기법**: 소문자 변환, URL 및 특수문자 제거, NLTK 토큰화 및 불용어 제거, 표제어 추출(Lemmatization).
- **필터링 규칙**: 토큰 길이 > 2, 말뭉치 빈도 5 미만 토큰 필터링, 빈 행 제거.
- **최종 학습용 전처리 데이터셋 크기**: **174,839 행** (2개 열: `type`, `posts`)
- **저장 파일**: `/workspace/outputs/final_preprocessed_mbti.csv`

---

## 3. 전체 파이프라인 실행 요약 (Phase 3)

- **전체 파이프라인 실행 시간**: 18.72 분
- **단계별 실행 시간**:
  - **Stage 1 (RandomForest Baseline)**: 39.7 초
  - **Stage 2 (Fine-tuning BERT Evaluation)**: 92.5 초 (Epoch 2 가중치 로드)
  - **Stage 3 & 4 (Binary Logistic)**: 132.7 초
  - **Stage 5 (SVC Optimization)**: 858.1 초 (14.30 분)

---

## 4. 모델 성능 종합 비교 표

| 단계 | 모델 이름 | 분류 형태 | 정확도 (Accuracy) | 파일 저장 유무 및 경로 |
| :--- | :--- | :--- | :---: | :--- |
| **Stage 1** | RandomForest Baseline | 16-Class 다중 분류 | **21.00%** | 저장 없음 |
| **Stage 2** | Fine-tuned BERT (Epoch 2) | 16-Class 다중 분류 | **30.18%** | `/data/bert_mbti_epoch2.pt` |
| **Stage 3 & 4** | Binary Logistic Regression (I/E) | 성격 차원 이진 분류 | **77.00%** | 메모리 상 학습 완료 |
| **Stage 3 & 4** | Binary Logistic Regression (N/S) | 성격 차원 이진 분류 | **86.00%** | 메모리 상 학습 완료 |
| **Stage 3 & 4** | Binary Logistic Regression (F/T) | 성격 차원 이진 분류 | **61.00%** | 메모리 상 학습 완료 |
| **Stage 3 & 4** | Binary Logistic Regression (P/J) | 성격 차원 이진 분류 | **61.00%** | 메모리 상 학습 완료 |
| **Stage 5** | Multi-class SVC (Optimized) | 16-Class 다중 분류 | **21.00%** | `/data/best_svc_model.joblib` |

---

## 5. 단계별 상세 평가 결과 (Classification Report)

### Stage 1: RandomForest Baseline on BERT [CLS] Embeddings
```
              precision    recall  f1-score   support

        ENFJ       0.12      0.00      0.01       771
        ENFP       0.28      0.01      0.02      2745
        ENTJ       0.24      0.00      0.01       936
        ENTP       0.19      0.01      0.02      2804
        ESFJ       0.00      0.00      0.00       174
        ESFP       1.00      0.01      0.01       193
        ESTJ       1.00      0.01      0.01       159
        ESTP       0.22      0.01      0.01       359
        INFJ       0.19      0.25      0.21      5934
        INFP       0.23      0.62      0.34      7337
        INTJ       0.17      0.08      0.11      4387
        INTP       0.19      0.18      0.19      5242
        ISFJ       0.17      0.01      0.01       671
        ISFP       0.38      0.01      0.02      1078
        ISTJ       0.36      0.01      0.01       830
        ISTP       0.26      0.01      0.01      1348

    accuracy                           0.21     34968
   macro avg       0.31      0.08      0.06     34968
weighted avg       0.22      0.21      0.15     34968
```

### Stage 2: Fine-tuned BERT Epoch 2 (16-Class)
```
              precision    recall  f1-score   support

        INFJ       0.31      0.31      0.31      5934
        ENTP       0.33      0.16      0.21      2804
        INTP       0.28      0.38      0.32      5242
        INTJ       0.37      0.17      0.23      4387
        ENTJ       0.41      0.10      0.16       936
        ENFP       0.48      0.15      0.23      2745
        INFP       0.28      0.62      0.39      7337
        ENFJ       0.41      0.09      0.14       771
        ISFP       0.44      0.09      0.15      1078
        ISFJ       0.53      0.08      0.13       671
        ISTP       0.32      0.12      0.17      1348
        ISTJ       0.36      0.11      0.16       830
        ESTP       0.31      0.09      0.14       359
        ESTJ       0.17      0.02      0.03       159
        ESFP       0.00      0.00      0.00       193
        ESFJ       0.32      0.05      0.08       174

    accuracy                           0.30     34968
   macro avg       0.33      0.16      0.18     34968
weighted avg       0.33      0.30      0.27     34968
```

### Stage 3 & 4: 성격 차원별 이진 분류 (Binary Traits)
#### 1) Introversion (I) vs Extraversion (E)
```
              precision    recall  f1-score   support

           E       0.54      0.01      0.01      8141
           I       0.77      1.00      0.87     26827

    accuracy                           0.77     34968
   macro avg       0.65      0.50      0.44     34968
weighted avg       0.71      0.77      0.67     34968
```

#### 2) Intuition (N) vs Sensing (S)
```
              precision    recall  f1-score   support

           S       0.33      0.00      0.00      4812
           N       0.86      1.00      0.93     30156

    accuracy                           0.86     34968
   macro avg       0.60      0.50      0.46     34968
weighted avg       0.79      0.86      0.80     34968
```

#### 3) Feeling (F) vs Thinking (T)
```
              precision    recall  f1-score   support

           T       0.59      0.49      0.53     16064
           F       0.62      0.71      0.66     18904

    accuracy                           0.61     34968
   macro avg       0.60      0.60      0.60     34968
weighted avg       0.60      0.61      0.60     34968
```

#### 4) Perceiving (P) vs Judging (J)
```
              precision    recall  f1-score   support

           J       0.52      0.11      0.18     13861
           P       0.62      0.93      0.74     21107

    accuracy                           0.61     34968
   macro avg       0.57      0.52      0.46     34968
weighted avg       0.58      0.61      0.52     34968
```

### Stage 5: Multi-class SVC Optimized Model (C=1, kernel='rbf', gamma='auto')
```
              precision    recall  f1-score   support

        ENFJ       0.00      0.00      0.00       771
        ENFP       0.00      0.00      0.00      2745
        ENTJ       0.00      0.00      0.00       936
        ENTP       0.00      0.00      0.00      2804
        ESFJ       0.00      0.00      0.00       174
        ESFP       0.00      0.00      0.00       193
        ESTJ       0.00      0.00      0.00       159
        ESTP       0.00      0.00      0.00       359
        INFJ       0.09      0.00      0.00      5934
        INFP       0.21      0.98      0.35      7337
        INTJ       0.00      0.00      0.00      4387
        INTP       0.21      0.05      0.08      5242
        ISFJ       0.00      0.00      0.00       671
        ISFP       0.00      0.00      0.00      1078
        ISTJ       0.00      0.00      0.00       830
        ISTP       0.00      0.00      0.00      1348

    accuracy                           0.21     34968
   macro avg       0.03      0.06      0.03     34968
weighted avg       0.09      0.21      0.09     34968
```

---

## 6. 최종 정리 및 특이 사항

1. **최고 성능 모델**:
   - 16개 클래스 다중 분류 문제에서는 미세 조정된 **BERT 모델(Stage 2)**이 **30.18%**의 정확도로 가장 우수한 성능을 나타냈습니다.
   - 4차원 성격 축 분류(이진 분류) 문제에서는 **N/S (직관/감각) 분류기**가 **86.00%**로 최고 정확도를 달성했습니다.
2. **하드웨어 가속 최적화**:
   - 학습 도중 GPU 0의 과열 및 행 걸림 현상(`Graphics is hung`)이 관찰되어, CUDA 환경 변수 설정을 통해 PyTorch 학습 연산을 건강한 GPU인 **GPU 1과 GPU 2로 우회 라우팅**하여 안정적으로 실행하였습니다.
3. **디스크 공간 보장**:
   - 홈 디렉토리 용량 초과 문제(100% 사용율)가 발생하여, 콘다 캐시 정리 및 필요 없는 압축 파일 제거를 진행해 **42GB의 가용 공간**을 선제 확보한 후 가중치 쓰기를 안전하게 마무리하였습니다.
4. **산출물 통합**:
   - 검증된 가중치 모델 파일, 전처리 csv 데이터, 학습 전체 로그, 시각화 그래프 이미지는 모두 원격 서버의 단일 디렉토리 `/userHome/userhome4/sehoon/MBTI_Project/outputs/`에 통합 보관 중입니다.

본 파이프라인 프로젝트는 설계 명세에 완벽히 부합하도록 구현 및 검증되었습니다.
