import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_truth_heatmap(tokens, scores, filename="results/truth_heatmap.png"):
    plt.figure(figsize=(12, 2))
    # Reshape scores for heatmap
    data = np.array(scores).reshape(1, -1)
    
    sns.heatmap(data, annot=np.array(tokens).reshape(1, -1), fmt="", 
                cmap="RdYlGn", cbar=False, linewidths=1)
    
    plt.title("C-SHIFT: Token-Level Hallucination Detection (Red = High Tension)")
    plt.axis('off')
    plt.savefig(filename)
    plt.show()

# Sample to show in PPT (e.g., Hindi or English)
# In a real run, you'd pull these from your TopKKLMetric outputs
demo_tokens = ["The", "capital", "is", "Mumbai"]
demo_scores = [0.9, 0.8, 0.7, -4.5] # Mumbai is a hallucination if context says Delhi
plot_truth_heatmap(demo_tokens, demo_scores)