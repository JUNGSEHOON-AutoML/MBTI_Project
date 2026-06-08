import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def extract_traits(labels):
    # Labels are 4-letter strings (e.g. 'INFJ')
    # Returns 4 lists representing: I/E, N/S, F/T, P/J
    # 1 if first trait (I, N, F, P), 0 if second trait (E, S, T, J)
    ie = np.array([1 if l[0] == 'I' else 0 for l in labels])
    ns = np.array([1 if l[1] == 'N' else 0 for l in labels])
    ft = np.array([1 if l[2] == 'F' else 0 for l in labels])
    pj = np.array([1 if l[3] == 'P' else 0 for l in labels])
    return ie, ns, ft, pj

def main():
    print("==================================================")
    print("Stage 3 & 4: Binary Logistic Regressions on Traits")
    print("==================================================")
    
    emb_path = '/data/bert_embeddings.npy'
    label_path = '/data/labels.npy'
    
    if not os.path.exists(emb_path) or not os.path.exists(label_path):
        print(f"ERROR: Embeddings or labels missing. Run extract_embeddings.py first.")
        sys.exit(1)
        
    print("Loading embeddings and labels...")
    X = np.load(emb_path)
    y_labels = np.load(label_path)
    
    # Extract traits
    ie, ns, ft, pj = extract_traits(y_labels)
    traits = {
        "I/E (Introversion vs Extraversion)": ie,
        "N/S (Intuition vs Sensing)": ns,
        "F/T (Feeling vs Thinking)": ft,
        "P/J (Perceiving vs Judging)": pj
    }
    
    # Train and evaluate independent models
    for trait_name, y in traits.items():
        print(f"\n--- Training Binary Model for {trait_name} ---")
        
        # Train-test split (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        # Target names corresponding to 0 and 1
        # Trait mappings:
        # I/E: 0 = E, 1 = I
        # N/S: 0 = S, 1 = N
        # F/T: 0 = T, 1 = F
        # P/J: 0 = J, 1 = P
        trait_code = trait_name.split()[0].split('/')
        target_names = [trait_code[1], trait_code[0]] # index 0 is second char, index 1 is first char
        
        print(f"Classification Report for {trait_name}:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
    print("==================================================")

if __name__ == '__main__':
    main()
