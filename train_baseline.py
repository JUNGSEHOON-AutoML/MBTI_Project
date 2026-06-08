import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def main():
    print("==================================================")
    print("Stage 1 (Baseline): RandomForest on BERT Embeddings")
    print("==================================================")
    
    emb_path = '/data/bert_embeddings.npy'
    label_path = '/data/labels.npy'
    
    if not os.path.exists(emb_path) or not os.path.exists(label_path):
        print(f"ERROR: Embeddings or labels missing. Run extract_embeddings.py first.")
        sys.exit(1)
        
    print("Loading embeddings and labels...")
    X = np.load(emb_path)
    y = np.load(label_path)
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)
    
    print("Evaluating baseline model...")
    y_pred = clf.predict(X_test)
    
    print("\n--- Classification Report (Stage 1 - RandomForest Baseline) ---")
    print(classification_report(y_test, y_pred))
    print("==================================================")

if __name__ == '__main__':
    main()
