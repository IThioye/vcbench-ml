import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib
import argparse

# Import from local modules
from training_pipeline import (
    train_all, load_data, compare_models, 
    plot_feature_importance, plot_learning_curve,
    plot_metric_comparison, plot_confusion_matrix,
    best_f0_5_threshold,
    MODELS_DIR
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATA_PATH      = Path("data/vcbench_final_public.csv")
RANDOM_STATE   = 42
TEST_SIZE      = 0.20

def main():
    parser = argparse.ArgumentParser(description="Run Research/Benchmark Pipeline")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--mode", type=str, default="research", choices=["research", "benchmark"], help="Mode: research or benchmark")
    parser.add_argument("--private_data", type=str, default="data/vcbench_final_private.csv", help="Path to the private test set")
    parser.add_argument("--use_text", action="store_true", help="Use TF-IDF text features (default: True)")
    parser.add_argument("--no_text", action="store_false", dest="use_text", help="Disable text features")
    parser.set_defaults(use_text=True)
    args = parser.parse_args()
    
    n_trials = args.n_trials
    mode = args.mode
    use_text = args.use_text
    
    # Separate output directories
    output_dir = Path("benchmark_results") if mode == "benchmark" else Path("research_results")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Starting Research Pipeline [{mode.upper()} MODE]")
    print(f"Outputs will be saved to: {output_dir}")

    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found!")
        return

    raw_data = pd.read_csv(DATA_PATH).replace({np.nan: None})
    
    if mode == "research":
        # 2. Train/Test Split (Reproducible)
        print(f"Splitting data (Test size: {TEST_SIZE})...")
        train_data, test_data = train_test_split(
            raw_data, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, 
            stratify=raw_data["success"] if "success" in raw_data.columns else None
        )
        train_records = train_data.to_dict("records")
        test_records  = test_data.to_dict("records")
        print(f"   Train: {len(train_records)} samples")
        print(f"   Test:  {len(test_records)} samples")
    else:
        # Benchmark mode: Use all public data for training
        print(f"Benchmark mode: Using all {len(raw_data)} public samples for training.")
        train_records = raw_data.to_dict("records")
        
        private_data_path = Path(args.private_data)
        if not private_data_path.exists():
            print(f"Error: Private data not found at {private_data_path}")
            return
        test_data = pd.read_csv(private_data_path).replace({np.nan: None})
        test_records = test_data.to_dict("records")
        print(f"Loaded {len(test_records)} private samples for prediction.")
    
    # 3. Train all models with Optuna tuning
    print(f"Training models with Optuna tuning ({n_trials} trials per model)...")
    results = train_all(train_records, tune_hyperparams=True, n_trials=n_trials, use_text=use_text)
    
    if mode == "benchmark":
        run_benchmark_submission(results, test_records, output_dir)
        return
    
    # 4. Evaluate on test set
    print(f"Evaluating all models on held-out test set...")
    X_test, y_test, _ = load_data(test_records)
    
    comparison_df = compare_models(
        results, 
        X_test, 
        y_test, 
        save_path=str(output_dir / "model_comparison_summary.png")
    )
    
    # Save comparison table
    comparison_df.to_csv(output_dir / "comparison_metrics.csv")
    print(f"Comparison table saved to {output_dir / 'comparison_metrics.csv'}")

    # Plot metric comparison (Precision, Recall, F0.5)
    plot_metric_comparison(comparison_df, output_dir / "metrics_comparison")
    
    # 5. Generate detailed reports for EACH model
    print(f"Generating detailed diagnostic plots for all models...")
    for model_name, res in results.items():
        print(f"   Processing {model_name}...")
        pipeline = res["pipeline"]
        plot_feature_importance(pipeline, res["X"], model_name, output_dir=output_dir / "feature_importance")
        plot_learning_curve(pipeline, res["X"], res["y"], model_name, output_dir=output_dir / "learning_curve")
        plot_confusion_matrix(pipeline, X_test, y_test, model_name, output_dir=output_dir / "confusion_matrices")
        
    # 6. Identify and save the Best Model
    best_model_name = comparison_df["auc_pr"].idxmax()
    print(f"Best model found: {best_model_name}")
    
    best_pipeline = results[best_model_name]["pipeline"]
    model_save_path = output_dir / f"best_model-{best_model_name}.pkl"
    joblib.dump(best_pipeline, model_save_path)
    print(f"Best model saved to {model_save_path}")
    
    # 7. Save summary metadata
    summary = {
        "best_model": best_model_name,
        "n_samples_train": len(train_records),
        "n_samples_test": len(test_records),
        "random_state": RANDOM_STATE,
        "metrics": comparison_df.to_dict(orient="index")
    }
    with open(output_dir / "research_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nPipeline completed successfully! Assets in '{output_dir}'.")

def run_benchmark_submission(results, test_records, output_dir):
    """Generate a binary submission CSV for the private benchmark set."""
    print("\nGenerating benchmark submission...")
    from training_pipeline import load_data
    X_private, _, ids_private = load_data(test_records)
    
    # AUTOMATIC SELECTION: Pick best model based on CV Average Precision (test_mean)
    best_name = max(results.keys(), key=lambda k: results[k]["cv_summary"]["avg_prec"]["test_mean"])
    
    print(f"Automatically selected best model based on CV: '{best_name}'")
    best_res = results[best_name]
    best_pipeline = best_res["pipeline"]
    
    # Find the optimal threshold found during CV/Training (saved in results if possible)
    # We use best_f0_5_threshold on the training set to get the most balanced binary cutoff
    print(f"Finding optimal threshold for binary prediction...")
    threshold = best_f0_5_threshold(best_pipeline, best_res["X"], best_res["y"])
    
    # Predict probabilities then convert to explicit binary integers (0/1)
    probas = best_pipeline.predict_proba(X_private)[:, 1]
    y_pred = np.where(probas >= threshold, 1, 0)
    
    submission = pd.DataFrame({
        "founder_uuid": ids_private,
        "success": y_pred
    })
    
    # Save submission
    sub_file = output_dir / "submission.csv"
    submission.to_csv(sub_file, index=False)
    print(f"Success! Binary submission saved to {sub_file}")
    print(f"Optimal threshold used: {threshold:.4f}")
    
    # Save best model separately
    model_save_path = output_dir / f"best_model-{best_name}.pkl"
    joblib.dump(best_pipeline, model_save_path)
    print(f"Final model saved to {model_save_path}")

if __name__ == "__main__":
    main()
