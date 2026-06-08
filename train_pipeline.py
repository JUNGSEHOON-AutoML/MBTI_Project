import os
import sys
import time
import subprocess

def run_script(script_name):
    print(f"\n==================================================")
    print(f"Executing: {script_name}")
    print(f"==================================================")
    start_time = time.time()
    
    # Run the script using the current python executable
    res = subprocess.run([sys.executable, script_name], capture_output=False)
    
    elapsed = time.time() - start_time
    if res.returncode == 0:
        print(f"SUCCESS: {script_name} completed in {elapsed:.1f}s")
        return True, elapsed
    else:
        print(f"FAILED: {script_name} failed with exit code {res.returncode}")
        return False, elapsed

def main():
    print("==================================================")
    print("Starting MBTI Model Training Master Pipeline")
    print("==================================================")
    master_start = time.time()
    
    pipeline_steps = []
    
    # Check if BERT embeddings and labels already exist in the container
    if os.path.exists('/data/bert_embeddings.npy') and os.path.exists('/data/labels.npy'):
        print("Pre-extracted BERT embeddings and labels found at /data. Skipping extract_embeddings.py.")
    else:
        pipeline_steps.append("extract_embeddings.py")
        
    pipeline_steps.extend([
        "train_baseline.py",
        "train_finetune.py",
        "train_binary.py",
        "train_svc.py"
    ])
    
    runtimes = {}
    for step in pipeline_steps:
        success, duration = run_script(step)
        runtimes[step] = duration
        if not success:
            print(f"\nPipeline interrupted due to failure in {step}")
            sys.exit(1)
            
    master_elapsed = time.time() - master_start
    print("\n==================================================")
    print("MBTI Training Master Pipeline Finished Successfully!")
    print("==================================================")
    print(f"Total Pipeline Runtime: {master_elapsed/60:.2f} minutes")
    print("Execution time per step:")
    for step, duration in runtimes.items():
        print(f" - {step}: {duration:.1f}s ({duration/60:.2f}m)")
    print("==================================================")

if __name__ == '__main__':
    main()
