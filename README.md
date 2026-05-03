# VCBench: Predicting Founder Success with Hybrid Semantic-Numeric Features

This repository contains a predictive modeling pipeline designed for the **VCBench** competition. The project implements a state-of-the-art approach that combines traditional numeric financial indicators with semantic text features extracted from founder and startup profiles.

## Key Features

- **Hybrid Preprocessing**: Seamlessly integrates numeric data scaling with TF-IDF semantic mining of text summaries.
- **Automated Model Selection**: Programmatically identifies the best-performing architecture (XGBoost, LightGBM, Random Forest, Logistic Regression, or Stacking Ensemble) using Cross-Validation Average Precision.
- **Threshold Optimization**: Maximizes the **F0.5 score** by dynamically calculating the optimal probability threshold, prioritizing precision over recall for venture capital decision support.
- **Dual-Mode Execution**: 
    - **Research Mode**: Comprehensive diagnostics, learning curves, and feature importance analysis on a train/test split.
    - **Benchmark Mode**: Full-dataset training and binary prediction generation for competition submission.
- **Ablation Study**: Integrated scripts to quantify the performance lift provided by semantic features compared to numeric baselines.

## Project Structure

```text
├── data/                  # Datasets (Public and Private)
├── models/                # Local storage for trained .pkl models
├── research_results/      # Diagnostic plots, metrics, and ablation assets
├── benchmark_results/     # Final submission.csv and production models
├── training_pipeline.py   # Core model logic and CV evaluation
├── feature_engineering.py # Feature extraction and TF-IDF logic
├── run_research.py        # Main entry point (Research & Benchmark modes)
└── run_ablation_study.py  # Script for text vs numeric comparison
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/IThioye/vcbench-ml.git
   cd vcbench-ml
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you have `xgboost`, `lightgbm`, `optuna`, `scikit-learn`, `pandas`, and `matplotlib` installed.)*

## Usage

### 1. Research & Diagnostic Mode
Use this mode to evaluate model performance, check for overfitting, and view feature importances.
```bash
python run_research.py --mode research --n_trials 50
```

### 2. Benchmark Submission Mode
Use this mode to train on the entire public dataset and generate a binary `submission.csv` for the private benchmark.
```bash
python run_research.py --mode benchmark --n_trials 20
```

### 3. Running the Ablation Study
To compare the performance with and without semantic text features:
```bash
python run_ablation_study.py --n_trials 10
```

## Results

The pipeline generates a variety of visual assets in the `research_results/` folder, including:
- **Precision-Recall Curves**: To visualize the trade-off between model confidence and coverage.
- **Feature Importance**: Identifying which founder traits contribute most to predicted success.
- **Ablation Comparison**: Side-by-side bar charts showing the lift in F0.5, Precision, and Recall when using TF-IDF.

