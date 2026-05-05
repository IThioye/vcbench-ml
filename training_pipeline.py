"""
training_pipeline.py
───────────────────────
Trains Logistic Regression, Random Forest, and AdaBoost
on the founder-success dataset with Optuna tuning, shared CV evaluation,
and a head-to-head comparison report.
 
Usage
-----
    from training_pipeline import train_all, compare_models
 
    results = train_all(train_records, tune_hyperparams=True, n_trials=50)
    compare_models(results, X_test_raw, y_test)
"""
 
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
 
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score,
 precision_recall_curve, fbeta_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.base import BaseEstimator, TransformerMixin, clone
import optuna
from optuna.samplers import TPESampler
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
 
from feature_engineering import build_feature_dataframe
 
# ─────────────────────────────────────────────
# Constants & Styling
# ─────────────────────────────────────────────
 
TARGET       = "success"
ID_COL       = "founder_uuid"
CV_FOLDS     = 3
RANDOM_STATE = 42
MODELS_DIR   = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Publication-quality palette
RESEARCH_PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#8C564B", "#E377C2"]
 
 
# ─────────────────────────────────────────────
# Shared: target encoder (same as before)
# ─────────────────────────────────────────────
 
class IndustryTargetEncoder(BaseEstimator, TransformerMixin):
    """Target-encodes industry_raw → industry_encoded, then drops the raw column."""
 
    def __init__(self, smooth="auto"):
        self.smooth   = smooth
        self.encoder_ = None
 
    def fit(self, X: pd.DataFrame, y=None):
        if "industry_raw" in X.columns:
            self.encoder_ = TargetEncoder(smooth=self.smooth, random_state=RANDOM_STATE)
            self.encoder_.fit(X[["industry_raw"]], y)
        return self
 
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self.encoder_ is not None and "industry_raw" in X.columns:
            X["industry_encoded"] = self.encoder_.transform(X[["industry_raw"]])[:, 0]
        if "industry_raw" in X.columns:
            X = X.drop(columns=["industry_raw"])
        return X
 
 
# ─────────────────────────────────────────────
# Shared: data loading
# ─────────────────────────────────────────────
 
def load_data(records: list[dict]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    for rec in records:
        rec["_industry_raw"] = rec.get("industry", "")
    df = build_feature_dataframe(records)
    df["industry_raw"] = [r["_industry_raw"] for r in records]
    
    # Handle target if present
    if TARGET in df.columns:
        y = df[TARGET].astype(int)
    else:
        y = pd.Series([0] * len(df)) # Dummy y for inference
        
    ids = df[ID_COL] if ID_COL in df.columns else pd.Series(range(len(df)))
    X   = df.drop(columns=[TARGET, ID_COL], errors="ignore")
    return X, y, ids
 
 
# ─────────────────────────────────────────────
# Preprocessing Helper
# ─────────────────────────────────────────────

def get_hybrid_preprocessor(numeric_cols: list[str], use_text: bool = True, text_max_features: int = 5000):
    """Returns a ColumnTransformer for numeric (scaled) and optionally text (TF-IDF) features."""
    transformers = [("num",  StandardScaler(), numeric_cols)]
    if use_text:
        transformers.append(("text", TfidfVectorizer(max_features=text_max_features, stop_words="english", ngram_range=(1, 2)), "text_summary"))
        
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

def get_numeric_columns(X: pd.DataFrame) -> list[str]:
    """Identify numeric columns, including the future industry_encoded."""
    cols = [c for c in X.columns if c not in ["text_summary", "industry_raw"]]
    if "industry_raw" in X.columns:
        cols.append("industry_encoded")
    return cols

# ─────────────────────────────────────────────
# Per-model pipeline builders
# ───────────────────────────────────────────── 
 
def build_logreg(params: dict, X_sample: pd.DataFrame, use_text: bool = False, text_max_features: int = 5000) -> Pipeline:
    # Handle Optuna artifacts (e.g., penalty_saga)
    model_params = params.copy()
    if "penalty_saga" in model_params:
        model_params["penalty"] = model_params.pop("penalty_saga")
    
    num_cols = get_numeric_columns(X_sample)
    return Pipeline([
        ("enc",    IndustryTargetEncoder()),
        ("prep",   get_hybrid_preprocessor(num_cols, use_text=use_text, text_max_features=text_max_features)),
        ("model",  LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **model_params,
        )),
    ])

 
 
def build_rf(params: dict, X_sample: pd.DataFrame, use_text: bool = False, text_max_features: int = 5000) -> Pipeline:
    num_cols = get_numeric_columns(X_sample)
    return Pipeline([
        ("enc",   IndustryTargetEncoder()),
        ("prep",  get_hybrid_preprocessor(num_cols, use_text=use_text, text_max_features=text_max_features)),
        ("model", RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )),
    ])



def build_adaboost(params: dict, X_sample: pd.DataFrame, use_text: bool = False, text_max_features: int = 5000) -> Pipeline:
    num_cols = get_numeric_columns(X_sample)
    return Pipeline([
        ("enc",   IndustryTargetEncoder()),
        ("prep",  get_hybrid_preprocessor(num_cols, use_text=use_text, text_max_features=text_max_features)),
        ("model", AdaBoostClassifier(
            random_state=RANDOM_STATE,
            **params,
        )),
    ])

def build_svm(params: dict, X_sample: pd.DataFrame, use_text: bool = True, text_max_features: int = 5000) -> Pipeline:
    num_cols = get_numeric_columns(X_sample)
    return Pipeline([
        ("enc",   IndustryTargetEncoder()),
        ("prep",  get_hybrid_preprocessor(num_cols, use_text=use_text, text_max_features=text_max_features)),
        ("model", SVC(
            random_state=RANDOM_STATE,
            probability=True,
            **params,
        )),
    ])

def build_mlp(params: dict, X_sample: pd.DataFrame, use_text: bool = True) -> Pipeline:
    # Reconstruct hidden_layer_sizes from dynamic Optuna parameters if present
    model_params = params.copy()
    if "n_layers" in model_params:
        n_layers = model_params.pop("n_layers")
        layers = []
        for i in range(n_layers):
            layers.append(model_params.pop(f"n_units_l{i}", 64)) # Default if missing
        model_params["hidden_layer_sizes"] = tuple(layers)
    
    num_cols = get_numeric_columns(X_sample)
    return Pipeline([
        ("enc",   IndustryTargetEncoder()),
        ("prep",  get_hybrid_preprocessor(num_cols, use_text=use_text)),
        ("model", MLPClassifier(
            random_state=RANDOM_STATE,
            max_iter=1000,
            **model_params,
        )),
    ])
 
# ─────────────────────────────────────────────
# Per-model Optuna search spaces
# ─────────────────────────────────────────────
 

def logreg_search_space(trial: optuna.Trial, pos_weight: float) -> dict:
    solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
    if solver == "saga":
        penalty = trial.suggest_categorical("penalty_saga", ["l2", "l1"])
    else:
        penalty = "l2"
    return {
        "C":             trial.suggest_float("C", 1e-3, 100.0, log=True),
        "penalty":       penalty,
        "solver":        solver,
        "class_weight":  trial.suggest_categorical("class_weight", ["balanced", None]),
    }
 
 
def rf_search_space(trial: optuna.Trial, pos_weight: float) -> dict:
    return {
        "n_estimators":   trial.suggest_int("n_estimators", 100, 600),
        "max_depth":      trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf":trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features":   trial.suggest_float("max_features", 0.3, 1.0),
        "class_weight":   trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
    }


def adaboost_search_space(trial: optuna.Trial, pos_weight: float) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
    }

def svm_search_space(trial: optuna.Trial, pos_weight: float) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"]),
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }

def mlp_search_space(trial: optuna.Trial, pos_weight: float) -> dict:
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f"n_units_l{i}", 16, 128))
    
    return {
        "hidden_layer_sizes": tuple(layers),
        "activation": trial.suggest_categorical("activation", ["tanh", "relu"]),
        "solver": trial.suggest_categorical("solver", ["sgd", "adam"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
        "learning_rate": trial.suggest_categorical("learning_rate", ["constant", "adaptive"]),
    }
 
# ─────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────
 
MODEL_REGISTRY = {
    "logreg":    {"builder": build_logreg, "search": logreg_search_space,  "default_params": lambda pw: {
        "C": 1.0, "penalty": "l2", "solver": "lbfgs", "class_weight": "balanced",
    }},
    "random_forest": {"builder": build_rf, "search": rf_search_space,      "default_params": lambda pw: {
        "n_estimators": 300, "max_depth": 8, "class_weight": "balanced",
    }},
    "adaboost": {"builder": build_adaboost, "search": adaboost_search_space, "default_params": lambda pw: {
        "n_estimators": 100, "learning_rate": 0.1,
    }},
    "svm": {"builder": build_svm, "search": svm_search_space, "default_params": lambda pw: {
        "C": 1.0, "kernel": "rbf", "gamma": "scale",
    }},
    "mlp": {"builder": build_mlp, "search": mlp_search_space, "default_params": lambda pw: {
        "hidden_layer_sizes": (64, 32), "activation": "relu", "solver": "adam", "alpha": 0.0001,
    }}
}
 
 
# ─────────────────────────────────────────────
# Shared: CV evaluation
# ─────────────────────────────────────────────
 
def evaluate_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[dict, list[dict]]:
    """Evaluate with benchmark-aligned 3-fold stratified CV using fold-specific F0.5 thresholds."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = {
        "roc_auc": {"train": [], "test": []},
        "avg_prec": {"train": [], "test": []},
        "f0.5": {"train": [], "test": []},
        "precision": {"train": [], "test": []},
        "recall": {"train": [], "test": []},
    }

    fold_reports = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        pipe = clone(pipeline)
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        pipe.fit(X_tr, y_tr)

        tr_proba = pipe.predict_proba(X_tr)[:, 1]
        va_proba = pipe.predict_proba(X_va)[:, 1]

        thr = best_f0_5_threshold_from_proba(y_va, va_proba)
        tr_pred = (tr_proba >= thr).astype(int)
        va_pred = (va_proba >= thr).astype(int)

        fold_metrics["roc_auc"]["train"].append(roc_auc_score(y_tr, tr_proba))
        fold_metrics["roc_auc"]["test"].append(roc_auc_score(y_va, va_proba))
        fold_metrics["avg_prec"]["train"].append(average_precision_score(y_tr, tr_proba))
        fold_metrics["avg_prec"]["test"].append(average_precision_score(y_va, va_proba))

        fold_metrics["f0.5"]["train"].append(fbeta_score(y_tr, tr_pred, beta=0.5, zero_division=0))
        fold_metrics["f0.5"]["test"].append(fbeta_score(y_va, va_pred, beta=0.5, zero_division=0))
        fold_metrics["precision"]["train"].append(precision_score(y_tr, tr_pred, zero_division=0))
        fold_metrics["precision"]["test"].append(precision_score(y_va, va_pred, zero_division=0))
        fold_metrics["recall"]["train"].append(recall_score(y_tr, tr_pred, zero_division=0))
        fold_metrics["recall"]["test"].append(recall_score(y_va, va_pred, zero_division=0))

        fold_reports.append({
            "fold": fold_idx,
            "n_train": int(len(tr_idx)),
            "n_valid": int(len(va_idx)),
            "positive_rate_valid": round(float(y_va.mean()), 4),
            "threshold": round(float(thr), 6),
            "f0.5": round(float(fold_metrics["f0.5"]["test"][-1]), 4),
            "precision": round(float(fold_metrics["precision"]["test"][-1]), 4),
            "recall": round(float(fold_metrics["recall"]["test"][-1]), 4),
        })
    summary = {}
    for metric, vals in fold_metrics.items():
        summary[metric] = {
            "test_mean": round(float(np.mean(vals["test"])), 4),
            "test_std": round(float(np.std(vals["test"])), 4),
            "train_mean": round(float(np.mean(vals["train"])), 4),
        }
    return summary, fold_reports
 
 
# ─────────────────────────────────────────────
# Shared: find best threshold on test set
# ─────────────────────────────────────────────
 
def best_f0_5_threshold(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    proba = pipeline.predict_proba(X)[:, 1]
    prec, rec, thresholds = precision_recall_curve(y, proba)
    f0_5s = (1 + 0.5**2) * prec * rec / (0.5**2 * prec + rec + 1e-8)
    return float(thresholds[np.argmax(f0_5s[:-1])])

def best_f0_5_threshold_from_proba(y_true: pd.Series, proba: np.ndarray) -> float:
    """Find the probability threshold that maximizes F0.5 from scores."""
    prec, rec, thresholds = precision_recall_curve(y_true, proba)
    f0_5s = (1 + 0.5**2) * prec * rec / (0.5**2 * prec + rec + 1e-8)
    return float(thresholds[np.argmax(f0_5s[:-1])])

def best_f0_5_threshold_oof(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> float:
    """Compute an F0.5-optimal threshold using out-of-fold probabilities."""
    oof_proba = np.zeros(len(y), dtype=float)
    fold_reports = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
        pipe = clone(pipeline)
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof_proba[va_idx] = pipe.predict_proba(X.iloc[va_idx])[:, 1]
    return best_f0_5_threshold_from_proba(y, oof_proba)
 
 
# ─────────────────────────────────────────────
# Single-model training
# ─────────────────────────────────────────────
 
def train_single(
    model_name: str,
    records: list[dict],
    tune_hyperparams: bool = True,
    n_trials: int = 50,
    use_text: bool = True,
    text_max_features: int = 5000,
) -> dict:
    """
    Train one model. Returns a result dict with keys:
        pipeline, cv_summary, best_params, study, X, y
    """
    assert model_name in MODEL_REGISTRY, f"Unknown model: {model_name}"
    reg = MODEL_REGISTRY[model_name]
 
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()}")
    print(f"{'='*60}")
 
    X, y, _ = load_data(records)
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    print(f"  {len(X)} samples | class balance: {y.mean():.1%} positive | pos_weight~{pos_weight:.1f}")

 
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
 
    if tune_hyperparams:
        print(f"  Optuna search ({n_trials} trials) ...") 
        def objective(trial):
            params   = reg["search"](trial, pos_weight)
            pipeline = reg["builder"](params, X, use_text=use_text)
            fold_scores = []
            for tr_idx, va_idx in cv.split(X, y):
                pipe = clone(pipeline)
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
                pipe.fit(X_tr, y_tr)
                proba = pipe.predict_proba(X_va)[:, 1]
                thr = best_f0_5_threshold_from_proba(y_va, proba)
                y_pred = (proba >= thr).astype(int)
                fold_scores.append(fbeta_score(y_va, y_pred, beta=0.5))
            return float(np.mean(fold_scores))
 
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_STATE),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_params = study.best_params
        print(f"  Best F0.5 (CV): {study.best_value:.4f}")
    else:
        best_params = reg["default_params"](pos_weight)
        study = None
        print("  Using default params.")
 
    # CV evaluation
    pipeline   = reg["builder"](best_params, X, use_text=use_text)
    cv_summary, cv_folds = evaluate_cv(pipeline, X, y)
 
    print(f"\n  {'Metric':<12} {'Test mean':>10} {'+/-std':>8} {'Train mean':>12}")

    print(f"  {'-'*44}")
    for m, v in cv_summary.items():
        print(f"  {m:<12} {v['test_mean']:>10.4f} {v['test_std']:>8.4f} {v['train_mean']:>12.4f}")
 
    print("\n  Out-of-sample (validation) fold metrics")
    print(f"  {'Fold':<6} {'F0.5':>8} {'Prec':>8} {'Rec':>8}")
    print(f"  ----------------------------------")
    for fold in cv_folds:
        print(
            f"  {fold['fold']:<6} {fold['f0.5']:>8.4f} {fold['precision']:>8.4f} {fold['recall']:>8.4f}"
        )
    print(f'  Mean   {cv_summary["f0.5"]["test_mean"]:>8.4f} {cv_summary["precision"]["test_mean"]:>8.4f} {cv_summary["recall"]["test_mean"]:>8.4f}')

    # Final fit
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODELS_DIR / f"{model_name}.pkl")
    oof_threshold = best_f0_5_threshold_oof(pipeline, X, y, cv)
    with open(MODELS_DIR / f"{model_name}_threshold.json", "w") as f:
        json.dump({"threshold": oof_threshold}, f, indent=2)
    with open(MODELS_DIR / f"{model_name}_cv_report.json", "w") as f:
        json.dump({"summary": cv_summary, "folds": cv_folds}, f, indent=2)
    print(f"\n  Saved -> models/{model_name}.pkl")
    print(f"  Saved -> models/{model_name}_threshold.json (threshold={oof_threshold:.3f})")
    print(f"  Saved -> models/{model_name}_cv_report.json")

 
    return {
        "model_name":  model_name,
        "pipeline":    pipeline,
        "cv_summary":  cv_summary,
        "best_params": best_params,
        "study":       study,
        "X":           X,
        "y":           y,
        "threshold":   oof_threshold,
        "cv_folds":    cv_folds,
    }
 
 
# ─────────────────────────────────────────────
# Train all models
# ─────────────────────────────────────────────
 
def train_all(
    records: list[dict],
    models: list[str] = None,
    tune_hyperparams: bool = True,
    n_trials: int = 50,
    use_text: bool = True,
    text_max_features: int = 5000,
) -> dict[str, dict]:
    """
    Train all (or a subset of) models on the same records.
 
    Parameters
    ----------
    records          : list of raw founder dicts (training set only)
    models           : list of model names to train; None = all four
    tune_hyperparams : run Optuna for each model
    n_trials         : Optuna trials per model
 
    Returns
    -------
    dict mapping model_name → result dict
    """
    models = models or list(MODEL_REGISTRY.keys())
    results = {}
    for name in models:
        use_text = True if name in ["mlp", "svm"] else False 
        results[name] = train_single(
            name, records, tune_hyperparams, n_trials,
            use_text=use_text, text_max_features=text_max_features
        )

    if len(results) >= 2:
        from training_pipeline import add_stacking_ensemble
        results = add_stacking_ensemble(results, records)
    print("\n[Done] All models trained.")

    return results

def add_stacking_ensemble(results: dict, records: list[dict]):
    """
    Creates a StackingClassifier using the already trained models in 'results'.
    Adds a new 'ensemble_stack' entry to the results dictionary.
    """
    print(f"\n{'='*60}")
    print(f"  BUILDING STACKING ENSEMBLE")
    print(f"{'='*60}")
    
    X, y, _ = load_data(records)
    
    # We use the already fitted pipelines as base estimators
    estimators = [
        (name, res["pipeline"]) 
        for name, res in results.items() 
        if name not in ["ensemble_stack", "mlp", "svm"]
    ]
    
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=RANDOM_STATE),
        n_jobs=None # Avoid nested parallelization issues
    )
    
    print(f"  Fitting meta-learner...")
    stack.fit(X, y)
    
    # Evaluation
    cv_summary, cv_folds = evaluate_cv(stack, X, y)

    print(f"\n  {'Metric':<12} {'Test mean':>10} {'+/-std':>8} {'Train mean':>12}")

    print(f"  {'-'*44}")
    for m, v in cv_summary.items():
        print(f"  {m:<12} {v['test_mean']:>10.4f} {v['test_std']:>8.4f} {v['train_mean']:>12.4f}")
 
    print("\n  Out-of-sample (validation) fold metrics")
    print(f"  {'Fold':<6} {'F0.5':>8} {'Prec':>8} {'Rec':>8}")
    print(f"  ----------------------------------")
    for fold in cv_folds:
        print(
            f"  {fold['fold']:<6} {fold['f0.5']:>8.4f} {fold['precision']:>8.4f} {fold['recall']:>8.4f}"
        )
    print(f'  Mean   {cv_summary["f0.5"]["test_mean"]:>8.4f} {cv_summary["precision"]["test_mean"]:>8.4f} {cv_summary["recall"]["test_mean"]:>8.4f}')

    # Save
    joblib.dump(stack, MODELS_DIR / "ensemble_stack.pkl")
    print(f"\n  Saved -> models/ensemble_stack.pkl")
    
    results["ensemble_stack"] = {
        "model_name":  "ensemble_stack",
        "pipeline":    stack,
        "cv_summary":  cv_summary,
        "cv_folds":    cv_folds,
        "best_params": {"final_estimator": "LogisticRegression"},
        "study":       None,
        "X":           X,
        "y":           y,
    }
    
    return results
 
 
# ─────────────────────────────────────────────
# Head-to-head comparison on held-out test set
# ─────────────────────────────────────────────
 
def compare_models(
    results: dict[str, dict],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str = "model_comparison.png",
) -> pd.DataFrame:
    """
    Evaluate all trained pipelines on the held-out test set,
    find optimal thresholds, print a comparison table,
    and save a 2×2 comparison chart.
 
    Parameters
    ----------
    results  : output of train_all()
    X_test   : raw feature DataFrame (must include industry_raw)
    y_test   : ground-truth labels
    """
    rows = []
    probas = {}
 
    for name, res in results.items():
        pipeline = res["pipeline"]
        proba    = pipeline.predict_proba(X_test)[:, 1]
        probas[name] = proba
 
        threshold = res.get("threshold")
        if threshold is None:
            threshold_path = MODELS_DIR / f"{name}_threshold.json"
            if threshold_path.exists():
                with open(threshold_path, "r") as f:
                    threshold = float(json.load(f).get("threshold", 0.5))
            else:
                threshold = best_f0_5_threshold(pipeline, X_test, y_test)
        y_pred    = (proba >= threshold).astype(int)
 
        rows.append({
            "model":      name,
            "auc_roc":    round(roc_auc_score(y_test, proba),            4),
            "auc_pr":     round(average_precision_score(y_test, proba),  4),
            "f0.5":         round(fbeta_score(y_test, y_pred, beta=0.5),                4),
            "precision":  round(precision_score(y_test, y_pred),         4),
            "recall":     round(recall_score(y_test, y_pred),            4),
            "threshold":  round(threshold, 3),
        })
 
    comparison = pd.DataFrame(rows).set_index("model").sort_values("f0.5", ascending=False)
 
    print("\n" + "="*72)
    print("  MODEL COMPARISON - Held-out test set")

    print("="*72)
    print(comparison.to_string())
    print("="*72)
    print(f"  Winner (F0.5): {comparison['f0.5'].idxmax()}")

 
    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors    = RESEARCH_PALETTE
    names     = list(results.keys())
 
    # 1. AUC-PR bar
    ax = axes[0, 0]
    cv_ap = [results[n]["cv_summary"]["avg_prec"]["test_mean"] for n in names]
    cv_ap_std = [results[n]["cv_summary"]["avg_prec"]["test_std"] for n in names]
    test_ap = [comparison.loc[n, "auc_pr"] if n in comparison.index else 0 for n in names]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w/2, cv_ap,  w, yerr=cv_ap_std, capsize=4, label="CV (train)", color=colors, alpha=0.6)
    ax.bar(x + w/2, test_ap, w, label="Test", color=colors, alpha=0.95)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12)
    ax.set_ylim(0, 1.1); ax.set_title("AUC-PR: CV vs Test", fontweight="bold")
    ax.legend(); ax.set_ylabel("Avg Precision")
 
    # 2. Precision-Recall curves
    ax = axes[0, 1]
    from sklearn.metrics import precision_recall_curve
    for i, (name, proba) in enumerate(probas.items()):
        p, r, _ = precision_recall_curve(y_test, proba)
        ax.plot(r, p, color=colors[i % len(colors)], lw=2, label=f"{name} (AP={comparison.loc[name,'auc_pr']:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontweight="bold")
    ax.legend(fontsize=8)
 
    # 3. f0.5 / Precision / Recall bar
    ax = axes[1, 0]
    metrics = ["f0.5", "precision", "recall"]
    bar_w = 0.2
    for i, name in enumerate(names):
        vals = [comparison.loc[name, m] for m in metrics]
        xs   = np.arange(len(metrics)) + i * bar_w
        ax.bar(xs, vals, bar_w, label=name, color=colors[i % len(colors)], alpha=0.85)
    ax.set_xticks(np.arange(len(metrics)) + bar_w * 1.5)
    ax.set_xticklabels(metrics); ax.set_ylim(0, 1.1)
    ax.set_title("f0.5 / Precision / Recall @ best threshold", fontweight="bold")
    ax.legend(fontsize=8)
 
    # 4. Score distribution per model (positive class only)
    ax = axes[1, 1]
    for i, (name, proba) in enumerate(probas.items()):
        ax.hist(proba[y_test == 1], bins=20, alpha=0.55, label=name,
                color=colors[i % len(colors)], density=True)
    ax.set_xlabel("P(success)"); ax.set_ylabel("Density")
    ax.set_title("Predicted prob. — positive class", fontweight="bold")
    ax.legend(fontsize=8)
 
    plt.suptitle("Multi-Model Comparison - Founder Success", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n[Chart] Comparison chart saved -> {save_path}")

    plt.close()
 
    return comparison
 
def plot_results(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                 cv_summary: dict, study: optuna.Study, model_name: str = "Model"):
    """Generate a comprehensive training report with multiple diagnostic plots."""
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
 
    # ── 1. CV metric bar chart ───────────────────────────────────────────
    ax1     = fig.add_subplot(gs[0, 0])
    metrics = list(cv_summary.keys())
    means   = [cv_summary[m]["test_mean"] for m in metrics]
    stds    = [cv_summary[m]["test_std"]  for m in metrics]
    bars    = ax1.bar(metrics, means, yerr=stds, capsize=5,
                      color="#4C72B0", alpha=0.85, ecolor="black")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("CV Metrics (mean ± std)", fontweight="bold")
    ax1.set_ylabel("Score")
    for bar, mean in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{mean:.3f}", ha="center", va="bottom", fontsize=9)
 
    # ── 2. Train vs test (overfit check) ────────────────────────────────
    ax2   = fig.add_subplot(gs[0, 1])
    x_pos = np.arange(len(metrics))
    width = 0.35
    ax2.bar(x_pos - width / 2, [cv_summary[m]["train_mean"] for m in metrics],
            width, label="Train", color="#55A868", alpha=0.85)
    ax2.bar(x_pos + width / 2, [cv_summary[m]["test_mean"]  for m in metrics],
            width, label="Test",  color="#4C72B0", alpha=0.85)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Train vs Test Score", fontweight="bold")
    ax2.legend()
 
    # ── 3. Feature importance ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    try:
        model = pipeline.named_steps["model"]
        feature_names = pipeline.named_steps["enc"].transform(X.head(1)).columns.tolist()
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            importances = np.zeros(len(feature_names))
        
        fi = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(15)
        fi.plot(kind="barh", ax=ax3, color="#C44E52", alpha=0.85)
        ax3.set_title("Top 15 Feature Importances", fontweight="bold")
    except Exception as e:
        ax3.text(0.5, 0.5, f"Plot error: {e}", ha="center", va="center")

    # ── 4. Optuna optimisation history ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if study:
        trial_values  = [t.value for t in study.trials if t.value is not None]
        running_best  = pd.Series(trial_values).cummax()
        ax4.plot(trial_values,  alpha=0.4, label="Trial",       color="#4C72B0")
        ax4.plot(running_best,  lw=2,      label="Best so far", color="#C44E52")
        ax4.set_title("Optuna Optimisation History", fontweight="bold")
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "No tuning history", ha="center", va="center")
 
    # ── 5. Hyperparameter importance ─────────────────────────────────────
    ax5    = fig.add_subplot(gs[1, 1])
    try:
        if study:
            imp    = optuna.importance.get_param_importances(study)
            top8   = dict(list(imp.items())[:8])
            params = list(top8.keys())
            vals   = list(top8.values())
            ax5.barh(params[::-1], vals[::-1], color="#8172B2", alpha=0.85)
            ax5.set_title("Hyperparameter Importance", fontweight="bold")
        else:
            ax5.text(0.5, 0.5, "No tuning history", ha="center", va="center")
    except Exception:
        ax5.text(0.5, 0.5, "Not enough trials", ha="center", va="center")
 
    # ── 6. Prediction distribution ───────────────────────────────────────
    ax6   = fig.add_subplot(gs[1, 2])
    proba = pipeline.predict_proba(X)[:, 1]
    ax6.hist(proba[y == 0], bins=30, alpha=0.6, label="Class 0", color="#4C72B0")
    ax6.hist(proba[y == 1], bins=30, alpha=0.6, label="Class 1", color="#C44E52")
    ax6.set_title("Predicted Probability Distribution", fontweight="bold")
    ax6.legend()
 
    plt.suptitle(f"{model_name} Training Report", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{model_name.lower().replace(' ', '_')}_training_report.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_feature_importance(pipeline: Pipeline, X: pd.DataFrame,
                             model_name: str = "Model", top_n: int = 15, 
                             output_dir: Path = Path("research_results")):
    """Extract feature importances (gain/coefs) and plot the top N."""
    output_dir.mkdir(exist_ok=True)
    try:
        # Check if this is a standard Pipeline or a raw model (like StackingClassifier)
        if hasattr(pipeline, "named_steps"):
            model = pipeline.named_steps["model"]
            # Correctly extract feature names from the ColumnTransformer (prep step)
            try:
                feature_names = pipeline.named_steps["prep"].get_feature_names_out()
            except:
                feature_names = [f"f{i}" for i in range(model.n_features_in_)]
        else:
            # For stacking ensembles or raw models, we skip feature importance for now
            # as it's not directly comparable to base models.
            return

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            suffix = "(Importance)"
        elif hasattr(model, "coef_"):
            # Handle sparse coefficients (common with SVM + TF-IDF)
            importances = model.coef_
            if hasattr(importances, "toarray"):
                importances = importances.toarray()
            importances = np.abs(importances[0]) if len(importances.shape) > 1 else np.abs(importances)
            suffix = "(Abs Coef)"
        else: return

        if len(importances) != len(feature_names):
            print(f"Warning: Importance length ({len(importances)}) != Name length ({len(feature_names)}) for {model_name}")
            return

        fi = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(top_n)
        plt.figure(figsize=(10, 7))
        fi.plot(kind="barh", color=plt.cm.viridis(np.linspace(0.4, 0.8, len(fi))), edgecolor="black")
        plt.title(f"{model_name} Feature Importance {suffix}", fontweight="bold")
        plt.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name.lower().replace(' ', '_')}_feature_importance.png", dpi=150)
        plt.close()
    except Exception as e: print(f"Error plotting FI for {model_name}: {e}")


def plot_learning_curve(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                         model_name: str = "Model", cv: int = 5,
                         output_dir: Path = Path("research_results")):
    """Plot the learning curve."""
    output_dir.mkdir(exist_ok=True)
    try:
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline, X, y, cv=cv, scoring="average_precision",
            train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
        )
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), "-o", label="Train")
        plt.plot(train_sizes, np.mean(test_scores, axis=1), "-o", label="CV")
        plt.title(f"Learning Curve: {model_name}", fontweight="bold")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f"{model_name.lower().replace(' ', '_')}_learning_curve.png", dpi=150)
        plt.close()
    except Exception as e: print(f"Error plotting LC: {e}")

def plot_metric_comparison(comparison_df: pd.DataFrame, output_dir: Path = Path("research_results")):
    """Plot individual bar charts for Precision, Recall, and F0.5 comparison and save them as separate files."""
    output_dir.mkdir(exist_ok=True)
    metrics = ["precision", "recall", "f0.5"]
    
    # Define stable color mapping for models
    all_models = comparison_df.index.tolist()
    model_colors = {name: RESEARCH_PALETTE[i % len(RESEARCH_PALETTE)] for i, name in enumerate(all_models)}
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Sort values but keep track of model names for colors
        data = comparison_df[metric].sort_values(ascending=False)
        bar_colors = [model_colors[name] for name in data.index]
        
        ax = data.plot(kind="bar", color=bar_colors, alpha=0.8, edgecolor="black")
        plt.title(f"Model Comparison: {metric.capitalize()}", fontweight="bold", fontsize=14, pad=15)
        plt.ylabel("Score", fontsize=12)
        plt.xlabel("Model", fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.xticks(rotation=45, ha="right")
        
        # Add values on top of bars
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha="center", va="center", xytext=(0, 9), textcoords="offset points", fontsize=10, fontweight="bold")
        
        plt.tight_layout()
        filename = output_dir / f"comparison_{metric}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"[Chart] Saved {filename}")
        plt.close()



def plot_confusion_matrix(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                          model_name: str = "Model", output_dir: Path = Path("research_results")):
    """Plot the confusion matrix at the best F0.5 threshold."""
    output_dir.mkdir(exist_ok=True)
    from training_pipeline import best_f0_5_threshold
    
    # Get optimal threshold
    threshold = best_f0_5_threshold(pipeline, X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    
    cm = confusion_matrix(y, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Failure", "Success"])
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    
    plt.title(f"Confusion Matrix: {model_name}\n(Threshold={threshold:.3f})", fontweight="bold")
    plt.tight_layout()
    
    filename = output_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"[Chart] Saved {filename}")
    plt.close()


# ─────────────────────────────────────────────
# Inference helper (any saved model)
# ─────────────────────────────────────────────
 
def predict(records: list[dict], model_name: str = "logreg") -> pd.DataFrame:
    """Load a saved pipeline and return predictions for new records."""
    pipeline = joblib.load(MODELS_DIR / f"{model_name}.pkl")
    X, _, ids = load_data(records)
    proba = pipeline.predict_proba(X)[:, 1]
    preds = pipeline.predict(X)
    return pd.DataFrame({
        "founder_uuid":      ids.values,
        "p_success":         np.round(proba, 4),
        "predicted_success": preds,
    })