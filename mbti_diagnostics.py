import os
import zipfile
import requests
import io
import re
import pandas as pd
from tabulate import tabulate

def download_file_from_google_drive(file_id, destination):
    print(f"Downloading from Google Drive ID: {file_id} to {destination}...")
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Check if virus scan warning is present and parse confirm token
    confirm_url = "https://drive.usercontent.google.com/download"
    confirm_match = re.search(r'name="confirm"\s+value="([^"]+)"', response.text)
    uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', response.text)
    
    if confirm_match:
        confirm_val = confirm_match.group(1)
        uuid_val = uuid_match.group(1) if uuid_match else ''
        print(f"Warning page detected. confirm={confirm_val}, uuid={uuid_val}")
        params = {'id': file_id, 'export': 'download', 'confirm': confirm_val}
        if uuid_val:
            params['uuid'] = uuid_val
        response = session.get(confirm_url, params=params, stream=True)
    else:
        # Fallback to direct confirm=t
        print("No warning page found or token parsed, attempting confirm=t...")
        response = session.get(URL, params={'id': file_id, 'confirm': 't'}, stream=True)
        
    try:
        with open(destination, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunk
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (50 * 1024 * 1024) == 0:
                        print(f"Downloaded {downloaded / (1024*1024):.1f} MB...")
        print(f"Download finished. Total size: {downloaded / (1024*1024):.1f} MB")
    finally:
        response.close()
        session.close()

def extract_zip(zip_path, extract_to="."):
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print("Extraction finished.")
            return True
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is not a valid zip file.")
            return False
    return False

def analyze_large_csv(csv_path):
    print(f"Memory-optimized analysis for {csv_path}...")
    # 1. Read head
    df_head = pd.read_csv(csv_path, nrows=5, on_bad_lines='skip')
    head_str = tabulate(df_head, headers='keys', tablefmt='pipe', showindex=False)
    
    # 2. Get columns and data types
    df_sample = pd.read_csv(csv_path, nrows=100, on_bad_lines='skip')
    dtypes = df_sample.dtypes
    
    # Initialize counts
    total_rows = 0
    missing_counts = {col: 0 for col in df_sample.columns}
    
    # 3. Process in chunks to prevent OOM
    chunk_size = 1000
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, on_bad_lines='skip'):
        total_rows += len(chunk)
        for col in chunk.columns:
            missing_counts[col] += chunk[col].isnull().sum()
            
    # Construct info() string equivalent
    info_lines = []
    info_lines.append(f"<class 'pandas.core.frame.DataFrame'>")
    info_lines.append(f"RangeIndex: {total_rows} entries, 0 to {total_rows - 1}")
    info_lines.append(f"Data columns (total {len(df_sample.columns)} columns):")
    info_lines.append(f" #   Column  Non-Null Count  Dtype")
    info_lines.append(f"---  ------  --------------  -----")
    for idx, col in enumerate(df_sample.columns):
        non_null = total_rows - missing_counts[col]
        info_lines.append(f" {idx:<3} {col:<7} {non_null:<15} {dtypes[col]}")
    info_lines.append(f"dtypes: {', '.join([f'{k}({v})' for k, v in dtypes.value_counts().items()])}")
    info_lines.append(f"memory usage: [Optimized Chunk Processing]")
    info_str = "\n".join(info_lines)
    
    # Missing summary
    missing_data = []
    for col in df_sample.columns:
        m_count = missing_counts[col]
        m_pct = (m_count / total_rows) * 100 if total_rows > 0 else 0
        missing_data.append([col, m_count, m_pct])
    df_missing = pd.DataFrame(missing_data, columns=["Column", "Missing Values", "Percentage (%)"])
    missing_str = tabulate(df_missing, headers='keys', tablefmt='pipe', showindex=False)
    
    return info_str, head_str, missing_str, total_rows, len(df_sample.columns)

def main():
    # 1. Unzip raw Kaggle MBTI dataset
    raw_zip = os.path.join("MBTI_Personality_Prediction", "mbti_1.csv.zip")
    if not os.path.exists("mbti_1.csv"):
        if not extract_zip(raw_zip):
            print("Error: mbti_1.csv.zip not found under MBTI_Personality_Prediction!")
            if os.path.exists("mbti_1.csv.zip"):
                extract_zip("mbti_1.csv.zip")
    
    # 2. Download preprocessed MBTI 500 dataset from Google Drive
    drive_id = "1ahF29-DHnM3VrAcL0MjdUEivJxUNOTqF"
    mbti_500_file = "MBTI_500_downloaded"
    
    # Remove bad MBTI 500.csv (previously downloaded 2KB error file)
    if os.path.exists("MBTI 500.csv") and os.path.getsize("MBTI 500.csv") < 100000:
        print("Removing invalid small MBTI 500.csv...")
        os.remove("MBTI 500.csv")
        
    if not os.path.exists("MBTI 500.csv") and not os.path.exists("mbti_500.csv"):
        download_file_from_google_drive(drive_id, mbti_500_file)
        if zipfile.is_zipfile(mbti_500_file):
            extract_zip(mbti_500_file)
            for f in os.listdir("."):
                if f.endswith(".csv") and "500" in f:
                    os.rename(f, "MBTI 500.csv")
                    break
        else:
            if os.path.exists("MBTI 500.csv"):
                try:
                    os.remove("MBTI 500.csv")
                except Exception as e:
                    print(f"Warning: Could not remove existing file: {e}")
            try:
                os.rename(mbti_500_file, "MBTI 500.csv")
            except Exception as e:
                print(f"Rename failed: {e}. Trying copy + delete fallback...")
                import shutil
                shutil.copy(mbti_500_file, "MBTI 500.csv")
                try:
                    os.remove(mbti_500_file)
                except Exception as e2:
                    print(f"Warning: Could not remove temp file: {e2}")
            
    # Resolve actual file names
    raw_csv = "mbti_1.csv"
    processed_csv = "MBTI 500.csv" if os.path.exists("MBTI 500.csv") else "mbti_500.csv"
    
    if not os.path.exists(raw_csv):
        print("Error: Raw Kaggle MBTI dataset (mbti_1.csv) is missing.")
        return
    if not os.path.exists(processed_csv):
        print("Error: Preprocessed MBTI 500 dataset is missing.")
        return
        
    print("Loading and analyzing datasets...")
    raw_info, raw_head, raw_missing, raw_rows, raw_cols = analyze_large_csv(raw_csv)
    proc_info, proc_head, proc_missing, proc_rows, proc_cols = analyze_large_csv(processed_csv)
    
    # Generate diagnostic report
    report_content = f"""# MBTI Datasets Diagnostic Report

This report summarizes the diagnostic checks performed on the two MBTI datasets:
1. **Dataset 1 (Raw Kaggle MBTI)**: `mbti_1.csv`
2. **Dataset 2 (Preprocessed MBTI 500)**: `{processed_csv}`

---

## 1. Dataset 1: Kaggle MBTI (Raw)

### Column Structure & Info
```
{raw_info}
```

### Missing Values Summary
{raw_missing}

### First 5 Rows (Head)
{raw_head}

---

## 2. Dataset 2: MBTI 500 (Preprocessed)

### Column Structure & Info
```
{proc_info}
```

### Missing Values Summary
{proc_missing}

### First 5 Rows (Head)
{proc_head}
"""

    report_path = "diagnostic_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Diagnostic report successfully written to {report_path}")
    print("\n--- Diagnostic Summary ---")
    print(f"Raw Kaggle MBTI rows: {raw_rows}, columns: {raw_cols}")
    print(f"MBTI 500 rows: {proc_rows}, columns: {proc_cols}")

if __name__ == "__main__":
    main()

