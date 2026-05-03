# Research Pipeline: Reproducibility & Publication

This directory contains the code used to train and evaluate models for founder success prediction. It is organized to facilitate easy reproduction of results and generation of high-quality assets for research papers.

## 📁 Key Files

- `run_research.py`: The main script to reproduce all results. It handles data splitting, hyperparameter tuning, model training, and asset generation.
- `training_pipeline.py`: Core library containing model definitions, feature processing, and plotting logic.
- `inference.py`: A clean utility to load the best-performing model and run predictions on new data.
- `research_results/`: (Generated) This folder will contain all your paper assets once you run the pipeline.

## 🚀 How to Reproduce Results

1. **Install Dependencies**:
   Ensure you have the required packages:
   ```bash
   pip install pandas numpy scikit-learn matplotlib xgboost lightgbm optuna joblib
   ```

2. **Run the Pipeline**:
   You can run the script in two modes:

   **A. Research Mode (Default)**
   Splits public data to generate diagnostic plots and metrics.
   ```bash
   python run_research.py --n_trials 50
   ```

   **B. Benchmark Mode**
   Trains on ALL public data to generate a `submission.csv` for VCBench.
   ```bash
   python run_research.py --mode benchmark --private_data data/vcbench_final_private.csv --n_trials 100
   ```

3. **Check the Output**:
   The `research_results/` folder will be created with:
   - `comparison_metrics.csv`: A table with AUC-ROC, AUC-PR, Precision, Recall, etc., for all models.
   - `best_model.pkl`: The top-performing pipeline, ready for deployment.
   - `*_feature_importance.png`: Detailed feature importance plots for each model.
   - `*_learning_curve.png`: Learning curves to analyze bias/variance for each model.
   - `model_comparison_summary.png`: A multi-panel chart comparing all models.

## 🔮 Running Inference

Once you have a trained "Best Model", you can use it in your own code or run the example script:
```bash
python inference.py
```

## 📝 Citation & Publication Note

When writing your paper:
- Use the tables in `comparison_metrics.csv` to build your Results section.
- The charts in `research_results/` are generated at 150 DPI with professional color palettes, suitable for direct insertion into documents.
- The use of **Optuna** ensures that you are presenting the best possible version of each model.
