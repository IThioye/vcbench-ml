import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from training_pipeline import train_all, compare_models, RESEARCH_PALETTE
from feature_engineering import build_feature_dataframe

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH = Path("data/vcbench_final_public.csv")
OUTPUT_DIR = Path("research_results/ablation_study")
N_TRIALS = 2  # Small number for quick demo
MODELS = ["logreg", "xgboost", "random_forest"]

def run_ablation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df_raw = pd.read_csv(DATA_PATH)
    # Convert back to list of dicts as training_pipeline expects
    records = df_raw.to_dict("records")
    
    # Split records for consistent comparison
    from sklearn.model_selection import train_test_split
    train_recs, test_recs = train_test_split(records, test_size=0.2, random_state=42)

    # Prepare test data
    from training_pipeline import load_data
    X_test, y_test, _ = load_data(test_recs)

    # 1. BASELINE (Numeric Only)
    print("\n" + "="*40)
    print("RUNNING BASELINE (Numeric Only)")
    print("="*40)
    results_base = train_all(train_recs, models=MODELS, n_trials=N_TRIALS, use_text=False)
    metrics_base = compare_models(results_base, X_test, y_test)
    metrics_base["mode"] = "Baseline"
    
    # 2. HYBRID (Numeric + Text)
    print("\n" + "="*40)
    print("RUNNING HYBRID (Numeric + Text)")
    print("="*40)
    results_hybrid = train_all(train_recs, models=MODELS, n_trials=N_TRIALS, use_text=True)
    metrics_hybrid = compare_models(results_hybrid, X_test, y_test)
    metrics_hybrid["mode"] = "Hybrid"
    
    # 3. COMPARISON
    print("\nGenerating Ablation Comparison...")
    comparison = pd.concat([metrics_base, metrics_hybrid]).reset_index()
    comparison.to_csv(OUTPUT_DIR / "ablation_metrics.csv", index=False)
    
    plot_ablation_results(comparison)
    print(f"\n[Done] Ablation results saved to {OUTPUT_DIR}")

def plot_ablation_results(df):
    # Melt the dataframe to tidy format for plotting multiple metrics
    metrics = ["f0.5", "precision", "recall"]
    df_plot = df.melt(id_vars=["model", "mode"], value_vars=metrics, var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(15, 8))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(data=df_plot, x="model", y="Score", hue="mode", palette=["#95a5a6", "#e74c3c"])
    
    # We'll use facets if possible, but a grouped bar chart is clearer for side-by-side
    plt.title("Ablation Study: Numeric vs Hybrid (Text) Features", fontsize=16, fontweight="bold", pad=20)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.ylim(0, 1.1)
    
    # We want to distinguish metrics, so maybe subplots?
    plt.close() # Reset
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    for i, metric in enumerate(metrics):
        sns.barplot(data=df, x="model", y=metric, hue="mode", ax=axes[i], palette=["#95a5a6", "#3498db"])
        axes[i].set_title(f"{metric.upper()}", fontweight="bold")
        axes[i].set_ylim(0, df[metrics].max().max() * 1.2)
        
        # Add labels
        for p in axes[i].patches:
            axes[i].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha="center", va="center", xytext=(0, 9), textcoords="offset points", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ablation_comparison_metrics.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    run_ablation()
