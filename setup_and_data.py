import os
import sys

def main():
    print("==================================================")
    print("Phase 1: Environment Setup & Dataset Download")
    print("==================================================")
    
    # 1. Target directory inside container / workspace
    # Default is '/data' inside the container, but fallback to local 'data' if running outside container.
    target_dir = '/data'
    if not os.path.exists('/') or not os.access('/', os.W_OK):
        # Fallback to local workspace data folder if running locally
        target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
        
    os.makedirs(target_dir, exist_ok=True)
    print(f"Dataset destination directory: {target_dir}")
    
    # 2. Check Kaggle API Credentials
    # Kaggle package expects KAGGLE_USERNAME and KAGGLE_KEY in os.environ
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')
    
    # Check fallback 1: /root/.kaggle/kaggle.json (standard location in container)
    container_kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    if (not username or not key) and os.path.exists(container_kaggle_json):
        try:
            import json
            with open(container_kaggle_json, 'r') as f:
                creds = json.load(f)
                os.environ['KAGGLE_USERNAME'] = creds.get('username', '')
                os.environ['KAGGLE_KEY'] = creds.get('key', '')
                print(f"Loaded Kaggle credentials from container home: {container_kaggle_json}")
        except Exception as e:
            print(f"Error reading {container_kaggle_json}: {e}")
            
    # Check fallback 2: Current directory or parent directory kaggle.json
    local_kaggle_json = os.path.join(os.path.dirname(__file__), 'kaggle.json')
    parent_kaggle_json = os.path.join(os.path.dirname(__file__), '..', 'kaggle.json')
    
    for path in [local_kaggle_json, parent_kaggle_json]:
        if (not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY')) and os.path.exists(path):
            try:
                import json
                with open(path, 'r') as f:
                    creds = json.load(f)
                    os.environ['KAGGLE_USERNAME'] = creds.get('username', '')
                    os.environ['KAGGLE_KEY'] = creds.get('key', '')
                    print(f"Loaded Kaggle credentials from workspace file: {path}")
                    break
            except Exception as e:
                print(f"Error reading {path}: {e}")
                
    # final credential check
    username = os.environ.get('KAGGLE_USERNAME')
    key = os.environ.get('KAGGLE_KEY')
    
    if not username or not key:
        print("WARNING: Kaggle environment variables (KAGGLE_USERNAME, KAGGLE_KEY) are missing.")
        print("Kaggle API will look for the credentials file ~/.kaggle/kaggle.json.")
        print("If download fails, please set the variables or mount a valid kaggle.json file.")
        
    # Initialize and authenticate Kaggle API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("Kaggle API authenticated successfully.")
    except Exception as e:
        print(f"ERROR: Kaggle API authentication failed: {e}")
        print("Please check your API token credentials.")
        sys.exit(1)
        
    # 3. Download and Extract datasets
    datasets = [
        ('datasnaek/mbti-type', 'datasnaek/mbti-type (Dataset 1)'),
        ('zeyadkhalid/mbti-personality-types-500-dataset', 'zeyadkhalid/mbti-personality-types-500-dataset (Dataset 2)')
    ]
    
    for dataset_id, label in datasets:
        print(f"\nDownloading {label}...")
        try:
            # dataset_download_files automatically downloads and extracts if unzip=True
            api.dataset_download_files(dataset_id, path=target_dir, unzip=True, quiet=False)
            print(f"Successfully downloaded and extracted {label} to {target_dir}")
        except Exception as e:
            print(f"ERROR: Failed to download {label}: {e}")
            sys.exit(1)
            
    # 4. NLTK Data Download (Runtime download instead of Docker build stage)
    print("\nDownloading NLTK resources...")
    try:
        import nltk
        for res in ['punkt', 'stopwords', 'wordnet']:
            print(f"Downloading NLTK resource '{res}'...")
            nltk.download(res, quiet=True)
        print("NLTK resources downloaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to download NLTK resources: {e}")
        sys.exit(1)

    # 5. GPU Sanity Check
    print("\n==================================================")
    print("GPU Sanity Check:")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            print(f"Current CUDA Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: CUDA is not available. PyTorch will use CPU.")
    except Exception as e:
        print(f"ERROR: Failed during GPU Sanity Check: {e}")
            
    print("\n==================================================")
    print("Phase 1 initialization complete!")
    print("==================================================")

if __name__ == '__main__':
    main()
