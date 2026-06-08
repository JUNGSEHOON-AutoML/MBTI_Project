# 텍스트 데이터 분석을 통한 MBTI 예측 (MBTI Personality Prediction using Text Data)

## 프로젝트 개요 (Project Overview)

- 주제: 텍스트 데이터 분석을 통한 MBTI 예측
- 배경: 기존 설문지 방식의 주관성과 시간 소요를 극복하기 위해, 사용자가 작성한 텍스트만으로 성격을 빠르고 객관적으로 예측하는 AI 서비스 개발

## 데이터셋 (Datasets)

- Kaggle MBTI Dataset: 전처리 전 데이터, 약 8,700개
- MBTI 500 Dataset: 전처리 완료 데이터, 약 106,000개

## 데이터 전처리 파이프라인 (Data Preprocessing)

- 1차 전처리: 소문자화, 정규식 기반 노이즈(URL, 기호) 제거, NLTK 토큰화 및 불용어 제거, WordNetLemmatizer 표제어 추출
- 2차 전처리: 두 데이터셋 병합 후 길이 2 이하 단어 제거, 코퍼스 내 등장 빈도 5 미만 단어 제거

## 모델링 및 성능 최적화 과정 (Modeling & Tuning)

입력 데이터는 bert-base-uncased 모델을 활용하여 문장 임베딩(768차원)을 추출해 사용합니다.

- Stage 1: Random Forest 기본 모델 (정확도 약 38%)
- Stage 2: BERT 시퀀스 분류기 파인튜닝 (정확도 약 85%)
- Stage 3: MBTI 4대 지표(I/E, N/S, F/T, P/J)를 분리한 Logistic Regression 이진 분류
- Stage 4 (최종): SVC 다중 분류기 파인튜닝 (GridSearchCV 결과 C=1, kernel='rbf', gamma='auto'에서 최종 테스트 정확도 90% 달성)

## 실행 방법 (How to Run)

Docker 환경 내부에서 아래의 스크립트를 순서대로 실행하여 전체 파이프라인을 구동합니다.

```bash
# 1. 환경 설정 및 데이터 다운로드
python setup_and_data.py

# 2. 데이터 병합 및 전처리 수행
python preprocess.py

# 3. 모델 학습 및 평가 파이프라인 실행
python train_pipeline.py
```
