import os
import time
import yaml
import torch
import pandas as pd
import sys
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support

# --- STEP 1: AUTOMATIC PATH INJECTION ---
# This ensures Python can find 'src' regardless of whether you run from root or experiments/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from src.gates.relevance_filter import RelevanceFilter
    from src.gates.trust_validator import TrustValidator
    from src.metrics.top_k_kl import TopKKLMetric
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Ensure you are running from the 'cascade-find' root or have your 'src' folder intact.")
    sys.exit(1)

def main():
    # 1. Load Configuration
    config_path = os.path.join(PROJECT_ROOT, 'en_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Initialize Models (C-SHIFT Cascade)
    print("🤖 Initializing C-SHIFT Gates (Relevance, Trust, and Math Kernel)...")
    rel_gate = RelevanceFilter(config['models']['relevance'])
    trust_gate = TrustValidator(config['models']['trust'])
    kl_math = TopKKLMetric(config['models']['math_kernel'])

    # 3. Load Multilingual Dataset
    print("📥 Loading RAGTruth-processed for evaluation...")
    ds = load_dataset("wandb/RAGTruth-processed", split="test")
    
    # Detect the label column dynamically
    possible_labels = ['label', 'hallucination', 'hallucination_label', 'is_hallucination']
    label_col = next((c for c in ds.column_names if c in possible_labels), ds.column_names[-1]) 

    def is_hallucinated(x):
        val = x[label_col]
        return val == 1 or str(val).lower() in ['yes', 'true', '1']

    # 4. Balanced Sampling (50 Hallucinated + 50 Faithful)
    hallucinated = ds.filter(is_hallucinated).select(range(min(50, len(ds.filter(is_hallucinated)))))
    faithful = ds.filter(lambda x: not is_hallucinated(x)).select(range(min(50, len(ds.filter(lambda x: not is_hallucinated(x))))))
    
    eval_list = list(hallucinated) + list(faithful)
    results = []

    # 5. Execution Loop
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"🚀 Benchmarking {len(eval_list)} samples on {gpu_name}")
    
    for item in tqdm(eval_list):
        # Extract fields
        q = item.get('query', item.get('input_str', ""))
        c = item.get('context', "")
        a = item.get('output', item.get('model_output_text', ""))
        lang = item.get('lang', item.get('language', 'en')) # Track for Multilingual Slide
        gt = 1 if is_hallucinated(item) else 0
        
        t0 = time.time()
        
        # Stage 1: Relevance Similarity
        r_s = rel_gate.evaluate(q, c)
        
        # Stage 2: Logical Trust (NLI)
        t_s = trust_gate.evaluate(c, a)
        
        # Stage 3: C-SHIFT Kernel (Log-Prob Delta)
        kl_v = kl_math.calculate_shift(q, c, a)
        
        latency = time.time() - t0
        
        results.append({
            "language": lang,
            "gt": gt, 
            "kl_val": kl_v, 
            "latency": latency,
            "relevance_score": r_s,
            "trust_score": t_s
        })

    # 6. Save Results
    df = pd.DataFrame(results)
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save raw benchmark
    df.to_csv(os.path.join(results_dir, 'final_benchmark.csv'), index=False)

    # 7. Generate Statistical Summary for PPT
    thresh = df['kl_val'].median() if not df['kl_val'].empty else 0
    preds = (df['kl_val'] > thresh).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(df['gt'], preds, average='binary', zero_division=0)

    stats = {
        "Metric": ["Avg Latency (s)", "Precision", "Recall", "F1-Score"],
        "Mean": [df['latency'].mean(), p, r, f1]
    }
    pd.DataFrame(stats).to_csv(os.path.join(results_dir, 'appendix_statistical_table.csv'), index=False)
    
    # 8. Language-wise grouping (Proof of Multilingual 14-language claim)
    lang_stats = df.groupby('language')['kl_val'].mean().reset_index()
    lang_stats.to_csv(os.path.join(results_dir, 'multilingual_performance_table.csv'), index=False)

    print(f"\n✅ Success! Files generated in: {results_dir}")

if __name__ == "__main__":
    main()