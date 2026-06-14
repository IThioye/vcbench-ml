# VCBench ML — Founder Success Prediction

This repository contains a **machine learning pipeline applied to founder/startup scoring** as part of the **VCBench** challenge.

The goal is to predict the binary `success` variable using structured data (experience, education, exits, industry) enriched by semantic signals (TF‑IDF on text summaries).

## Overview

The project provides:

* **Domain-specific feature engineering** from nested (JSON-like) fields;
* Multi-model training with **Optuna optimization**;
* Comparison across metrics tailored for imbalanced classification (AUC-PR, F0.5, precision, recall);
* Two execution modes:
* **research**: detailed analysis and diagnostics;
* **benchmark**: final training + `submission.csv` generation.



## Repository Structure

```text
├── data/                        # Public/private datasets (untracked depending on your setup)
├── models/                      # Serialized models (.pkl)
├── research_results/            # Diagnostic outputs (curves, tables, ablation)
├── benchmark_results/           # Submissions and artifacts from the benchmark mode
├── feature_engineering.py       # Construction of numerical features + text_summary
├── training_pipeline.py         # Training, CV, model selection and comparison
├── run_research.py              # Main entry point (research / benchmark)
├── run_ablation_study.py        # Baseline (numerical) vs hybrid (numerical + text) study
└── requirements.txt             # Python dependencies

```

## Expected Data

By default, the main script reads:

* `data/vcbench_final_public.csv` (train / public),
* `data/vcbench_final_private.csv` (private test for benchmark mode submission).

Important columns used by the pipeline:

* Identifier: `founder_uuid`
* Target (if available): `success`
* Raw fields for feature engineering: `industry`, `educations_json`, `jobs_json`, `ipos`, `acquisitions`

> In benchmark mode, the private file may not contain the `success` column (inference only).

## Installation

```bash
git clone https://github.com/IThioye/vcbench-ml.git
cd vcbench-ml
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

```

## How to Use the Repo

### 1) Run a full analysis (research mode)

This mode performs a train/test split, trains multiple models, compares performance, generates plots, and saves the best model.

```bash
python run_research.py --mode research --n_trials 50

```

Main outputs:

* `research_results/comparison_metrics.csv`
* `research_results/model_comparison_summary.png`
* Visualization folders (feature importance, learning curves, confusion matrices)
* `research_results/research_summary.json`
* `research_results/best_model-<model_name>.pkl`

### 2) Generate a submission (benchmark mode)

This mode trains on the entire public dataset, automatically selects the best model via CV (`f0.5`), computes an optimal threshold, and then generates a binary submission.

```bash
python run_research.py --mode benchmark --n_trials 20 --private_data data/vcbench_final_private.csv

```

Main outputs:

* `benchmark_results/submission.csv`
* `benchmark_results/cv_comparison_metrics.csv`
* `benchmark_results/best_model-<model_name>.pkl`

### 3) Enable/disable text features

Text features are enabled by default (`--use_text`). You can run a quick comparison using:

```bash
python run_research.py --mode research --n_trials 20 --no_text

```

### 4) Run the ablation study

Explicitly compares:

* **Baseline** = numerical variables only
* **Hybrid** = numerical + TF‑IDF

```bash
python run_ablation_study.py --n_trials 10

```

Outputs:

* `research_results/ablation_study/ablation_metrics.csv`
* `research_results/ablation_study/ablation_f0.5.png`
* `research_results/ablation_study/ablation_precision.png`
* `research_results/ablation_study/ablation_recall.png`

## Trained Models

The pipeline includes the following models:

* Logistic Regression
* Random Forest
* AdaBoost
* SVM
* MLP
* (and a stacking ensemble depending on configuration)

Hyperparameter optimization is handled by **Optuna**.

## Practical Tips

* Start with a low number of trials (`--n_trials 5` or `10`) to validate the pipeline.
* Increase to `30+` for more robust tuning.
* Use `--no_text` to measure the actual performance lift provided by semantic features on your current split.
