import os
import re
import sys
import time
import pandas as pd
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

# Worker function for parallel text cleaning (1차 전처리)
def clean_texts_batch(texts):
    # Setup NLTK resources inside the process worker
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    lemma_cache = {}

    cleaned_batch = []
    for text in texts:
        if not isinstance(text, str):
            cleaned_batch.append([])
            continue
        # 1. Lowercase and replace ||| with space
        text = text.lower().replace('|||', ' ')
        # 2. Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # 3. Remove special characters (keep only alphanumeric and spaces)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # 4. Tokenization
        tokens = word_tokenize(text)
        # 5. Stopwords & Lemmatization
        cleaned_tokens = []
        for t in tokens:
            if t not in stop_words:
                if t not in lemma_cache:
                    lemma_cache[t] = lemmatizer.lemmatize(t)
                cleaned_tokens.append(lemma_cache[t])
        cleaned_batch.append(cleaned_tokens)
    return cleaned_batch

def parallel_clean(texts, num_workers=4, chunk_size=500):
    total = len(texts)
    chunks = [texts[i:i + chunk_size] for i in range(0, total, chunk_size)]
    results = []
    
    print(f"Starting parallel preprocessing with {num_workers} workers on {total} items...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batches
        futures = [executor.submit(clean_texts_batch, chunk) for chunk in chunks]
        
        # Collect results
        for idx, future in enumerate(futures):
            results.extend(future.result())
            if (idx + 1) % 10 == 0 or (idx + 1) == len(chunks):
                elapsed = time.time() - start_time
                pct = (idx + 1) / len(chunks) * 100
                print(f"Processed {min((idx + 1) * chunk_size, total)}/{total} items ({pct:.1f}%) - Elapsed: {elapsed:.1f}s")
                
    return results

def main():
    print("==================================================")
    print("Phase 2: Data Merging & Text Preprocessing")
    print("==================================================")
    
    mbti_path = '/data/mbti_1.csv'
    mbti_500_path = '/data/MBTI 500.csv'
    output_path = '/data/final_preprocessed_mbti.csv'
    
    if not os.path.exists(mbti_path) or not os.path.exists(mbti_500_path):
        print(f"ERROR: Dataset files missing in /data. Please verify Phase 1 completed.")
        sys.exit(1)
        
    # 1. Load Datasets
    print("Loading datasets...")
    df_mbti = pd.read_csv(mbti_path)
    df_500 = pd.read_csv(mbti_500_path)
    
    print(f"Dataset 1 (mbti_1): {df_mbti.shape[0]} rows")
    print(f"Dataset 2 (MBTI 500): {df_500.shape[0]} rows")
    
    # Setup parallel workers count
    import multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    # 2. Preprocess Dataset 1 (mbti_1.csv) - 1차 전처리
    print("\n--- Preprocessing Dataset 1 (mbti_1) ---")
    df_mbti['tokens'] = parallel_clean(df_mbti['posts'].tolist(), num_workers=num_workers, chunk_size=500)
    
    # 3. Preprocess Dataset 2 (MBTI 500.csv) - 1차 전처리 완료 데이터이므로 단순 공백 분할(Tokenization)
    print("\n--- Extracting tokens from Dataset 2 (MBTI 500) ---")
    df_500['tokens'] = df_500['posts'].fillna('').apply(lambda x: str(x).split())
    
    # 4. Merge Datasets
    print("\nMerging Datasets...")
    df1 = pd.DataFrame({'type': df_mbti['type'], 'tokens': df_mbti['tokens']})
    df2 = pd.DataFrame({'type': df_500['type'], 'tokens': df_500['tokens']})
    df_merged = pd.concat([df1, df2], ignore_index=True)
    print(f"Merged corpus size: {df_merged.shape[0]} rows")
    
    # 5. Apply 2차 전처리 Filters
    print("\n--- Applying corpus-wide filters ---")
    
    # Filter A: Remove tokens of length <= 2
    print("Filtering out words of length <= 2...")
    df_merged['tokens'] = df_merged['tokens'].apply(lambda tokens: [t for t in tokens if len(t) > 2])
    
    # Filter B: Filter out tokens with corpus-wide frequency < 5
    print("Calculating word frequencies in the entire corpus...")
    word_counts = Counter()
    for tokens in df_merged['tokens']:
        word_counts.update(tokens)
        
    print(f"Total vocabulary size before frequency filter: {len(word_counts)}")
    frequent_words = {word for word, count in word_counts.items() if count >= 5}
    print(f"Vocabulary size after frequency filter (freq >= 5): {len(frequent_words)}")
    
    print("Filtering out rare words (freq < 5)...")
    df_merged['tokens'] = df_merged['tokens'].apply(lambda tokens: [t for t in tokens if t in frequent_words])
    
    # 6. Reconstruct cleaned posts string
    print("\nReconstructing text posts...")
    df_merged['posts'] = df_merged['tokens'].apply(lambda tokens: ' '.join(tokens))
    
    # Remove empty or whitespace-only rows
    before_drop = df_merged.shape[0]
    df_merged = df_merged[df_merged['posts'].str.strip() != '']
    after_drop = df_merged.shape[0]
    print(f"Dropped {before_drop - after_drop} empty rows after filtering.")
    
    # Save final results
    print(f"Saving final preprocessed dataset to {output_path}...")
    df_save = df_merged[['type', 'posts']]
    df_save.to_csv(output_path, index=False)
    print("Preprocessing completed successfully!")
    print(f"Final dataset shape: {df_save.shape}")
    
if __name__ == '__main__':
    main()
