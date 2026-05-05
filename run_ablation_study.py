import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from training_pipeline import train_all, compare_models, RESEARCH_PALETTE
from feature_engineering import build_feature_dataframe

import argparse

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH = Path("data/vcbench_final_public.csv")
OUTPUT_DIR = Path("research_results/ablation_study")
MODELS = ["logreg","random_forest","adaboost","svm","mlp"]

def run_ablation(n_trials=2):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Ablation Study (Trials: {n_trials})")
    print("Loading data...")
    df_raw = pd.read_csv(DATA_PATH)
    
    # Split records for consistent comparison WITH stratification
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df_raw, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_raw["success"] if "success" in df_raw.columns else None
    )
    
    train_recs = train_df.to_dict("records")
    test_recs  = test_df.to_dict("records")

    # Prepare test data
    from training_pipeline import load_data
    X_test, y_test, _ = load_data(test_recs)

    # 1. BASELINE (Numeric Only)
    print("\n" + "="*40)
    print("RUNNING BASELINE (Numeric Only)")
    print("="*40)
    results_base = train_all(train_recs, models=MODELS, n_trials=n_trials, use_text=False)
    metrics_base = compare_models(results_base, X_test, y_test)
    metrics_base["mode"] = "Baseline"
    
    # 2. HYBRID (Numeric + Text)
    print("\n" + "="*40)
    print("RUNNING HYBRID (Numeric + Text)")
    print("="*40)
    results_hybrid = train_all(train_recs, models=MODELS, n_trials=n_trials, use_text=True)
    metrics_hybrid = compare_models(results_hybrid, X_test, y_test)
    metrics_hybrid["mode"] = "Hybrid"
    
    # 3. COMPARISON
    print("\nGenerating Ablation Comparison...")
    comparison = pd.concat([metrics_base, metrics_hybrid]).reset_index()
    comparison.to_csv(OUTPUT_DIR / "ablation_metrics.csv", index=False)
    
    plot_ablation_results(comparison)
    print(f"\n[Done] Ablation results saved to {OUTPUT_DIR}")

def plot_ablation_results(df):
    metrics = ["f0.5", "precision", "recall"]
    
    sns.set_style("whitegrid")
    
    for metric in metrics:
        plt.figure(figsize=(12, 7))
        
        # Consistent color palette for Baseline vs Hybrid
        palette = {"Baseline": "#95a5a6", "Hybrid": "#3498db"}
        
        ax = sns.barplot(data=df, x="model", y=metric, hue="mode", palette=palette)
        
        plt.title(f"Ablation Study: {metric.upper()} Comparison", fontsize=16, fontweight="bold", pad=20)
        plt.ylabel(metric.capitalize(), fontsize=12)
        plt.xlabel("Model", fontsize=12)
        plt.ylim(0, df[metric].max() * 1.25)
        plt.xticks(rotation=30, ha="right")
        
        # Add labels on top of bars
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha="center", va="center", xytext=(0, 9), textcoords="offset points", fontsize=9, fontweight="bold")
        
        plt.legend(title="Feature Set", loc="upper right")
        plt.tight_layout()
        
        save_path = OUTPUT_DIR / f"ablation_{metric}.png"
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ablation Study")
    parser.add_argument("--n_trials", type=int, default=2, help="Number of Optuna trials")
    args = parser.parse_args()
    
    run_ablation(n_trials=args.n_trials)
