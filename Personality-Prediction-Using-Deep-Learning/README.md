# Personality Prediction Using Machine Learning and Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

This project investigates personality prediction from text using traditional machine learning (Naive Bayes) and advanced deep learning models (LSTM, Bi-LSTM, Bi-LSTM + Attention, DistilBERT). Based on the MBTI Personality Types 500 Dataset, it employs GloVe embeddings, SMOTE for class imbalance, and attention mechanisms. The Bi-LSTM + Attention model achieved a top accuracy of **96%** and validation loss of **13%**, highlighting the efficacy of deep learning for capturing linguistic and emotional cues in personality analysis.

Applications include personalized marketing, recruitment, mental health assessment, and recommender systems.

---

## Features

- **Dataset**: MBTI Personality Types 500 (~100,000 social media posts, 16 MBTI types).
- **Models**:
  - Naive Bayes (76% accuracy)
  - LSTM (70% accuracy)
  - Bi-LSTM (88% accuracy)
  - Bi-LSTM + Attention (96% accuracy, 13% loss)
  - DistilBERT (72% accuracy on subset)
- **Techniques**: GloVe embeddings, SMOTE, dropout, early stopping, attention mechanisms.
- **Code**: Jupyter Notebook with data exploration, model training, and evaluation.
- **Documentation**: Research paper detailing methodology and results.

---

## Repository Structure
personality-prediction-study/
├── datasets/                    
├── notebooks/
│   └── Personality Prediction Using Deep Learning.ipynb  
├── paper/
│   └── Personality Prediction Using Deep Learning.pdf 
├── README.md                    
└── requirements.txt             


---

## Installation

### Prerequisites
- Python 3.8+
- Google Colab (recommended for GPU) or local GPU setup

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/[your-username]/personality-prediction-study.git
   cd personality-prediction-study
2. **Install Dependencies**:
   ```bash
     pip install -r requirements.txt

### Dataset
The MBTI Personality Types 500 Dataset is used but not included due to size/licensing. Download it from [source link if available] or prepare your own MBTI data. The dataset is available at this link
(https://drive.google.com/file/d/1ahF29-DHnM3VrAcL0MjdUEivJxUNOTqF/view?usp=drive_link)

Place mbti_500.csv in datasets/.
Update the notebook path (/content/drive/MyDrive/Colab Notebooks/Datasets/mbti_500.csv) to your setup.

### Usage
1. Open the Notebook:
Upload personality_prediction.ipynb to Google Colab or run locally with Jupyter.
Ensure the dataset is accessible.

3. Run the Code:
Execute cells to preprocess data, train models, and view results.
Customize hyperparameters in the notebook if desired.

5. Explore Results:
Check accuracy/loss plots and metrics in the notebook.
Refer to the paper for detailed insights.

### Results
Model	Accuracy	Loss	Precision	Recall	F1-Score
Naive Bayes	0.76	-	0.77	0.76	0.77
LSTM	0.70	0.84	0.67	0.70	0.68
Bi-LSTM	0.88	0.41	0.88	0.88	0.88
Bi-LSTM + Attention	0.96	0.13	0.96	0.96	0.96
DistilBERT	0.72	0.94	0.73	0.72	0.72
Best Model: Bi-LSTM + Attention excels due to focused feature extraction.
Visuals: See notebook for training/validation plots.
