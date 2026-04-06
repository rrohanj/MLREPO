import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_all():
    df_stats = pd.read_csv('results/appendix_statistical_table.csv')
    your_lat = df_stats[df_stats['Metric'] == 'Latency (s)']['Mean'].values[0]
    your_f1 = df_stats[df_stats['Metric'] == 'F1-Score']['Mean'].values[0]

    data = {
        "Model": ["GPT-4o (Judge)", "Self-Check GPT", "REFIND (Original)", "Cascade-FIND (Ours)"],
        "F1-Score": [0.88, 0.74, 0.81, your_f1],
        "Latency (s)": [3.50, 5.20, 0.45, your_lat]
    }
    df = pd.DataFrame(data)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Model', y='F1-Score', data=df, ax=ax1, palette='mako', alpha=0.7)
    ax2 = ax1.twinx()
    sns.lineplot(x='Model', y='Latency (s)', data=df, marker='o', color='red', ax=ax2)
    plt.title('Cascade-FIND: Accuracy vs Latency')
    plt.savefig('results/superiority_comparison.png')
    plt.show()

if __name__ == "__main__":
    plot_all()
