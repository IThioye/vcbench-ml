# VCBench ML — Prédiction du succès des fondateurs

Ce dépôt contient un pipeline de **machine learning appliqué au scoring de fondateurs/startups** dans le cadre du challenge **VCBench**.  
L’objectif est de prédire la variable binaire `success` à partir de données structurées (expérience, éducation, exits, industrie) enrichies par des signaux sémantiques (TF‑IDF sur résumé texte).

## Vue d’ensemble

Le projet propose :

- un **feature engineering métier** à partir de champs imbriqués (JSON-like) ;
- un entraînement multi-modèles avec **optimisation Optuna** ;
- une comparaison sur des métriques orientées classification déséquilibrée (AUC-PR, F0.5, précision, rappel) ;
- deux modes d’exécution :
  - **research** : analyse détaillée et diagnostics ;
  - **benchmark** : entraînement final + génération de `submission.csv`.

## Structure du dépôt

```text
├── data/                        # Jeux public/privé (non versionnés selon votre setup)
├── models/                      # Modèles sérialisés (.pkl)
├── research_results/            # Sorties diagnostics (courbes, tableaux, ablation)
├── benchmark_results/           # Soumissions et artefacts du mode benchmark
├── feature_engineering.py       # Construction des features numériques + text_summary
├── training_pipeline.py         # Entraînement, CV, sélection et comparaison de modèles
├── run_research.py              # Point d’entrée principal (research / benchmark)
├── run_ablation_study.py        # Étude baseline (numérique) vs hybride (numérique + texte)
└── requirements.txt             # Dépendances Python
```

## Données attendues

Le script principal lit par défaut :

- `data/vcbench_final_public.csv` (train / public),
- `data/vcbench_final_private.csv` (test privé pour la soumission en mode benchmark).

Colonnes importantes utilisées par le pipeline :

- Identifiant : `founder_uuid`
- Cible (si disponible) : `success`
- Champs bruts pour feature engineering : `industry`, `educations_json`, `jobs_json`, `ipos`, `acquisitions`

> En mode benchmark, le fichier privé peut ne pas contenir `success` (inférence uniquement).

## Installation

```bash
git clone https://github.com/IThioye/vcbench-ml.git
cd vcbench-ml
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Comment utiliser le repo

### 1) Lancer une analyse complète (mode research)

Ce mode fait un split train/test, entraîne plusieurs modèles, compare les performances, génère des graphes et sauvegarde un meilleur modèle.

```bash
python run_research.py --mode research --n_trials 50
```

Sorties principales :

- `research_results/comparison_metrics.csv`
- `research_results/model_comparison_summary.png`
- dossiers de visualisations (importance, learning curves, confusion matrices)
- `research_results/research_summary.json`
- `research_results/best_model-<nom_modele>.pkl`

### 2) Générer une soumission (mode benchmark)

Ce mode entraîne sur tout le dataset public, sélectionne automatiquement le meilleur modèle via CV (`f0.5`), calcule un seuil optimal, puis produit une soumission binaire.

```bash
python run_research.py --mode benchmark --n_trials 20 --private_data data/vcbench_final_private.csv
```

Sorties principales :

- `benchmark_results/submission.csv`
- `benchmark_results/cv_comparison_metrics.csv`
- `benchmark_results/best_model-<nom_modele>.pkl`

### 3) Activer/désactiver les features texte

Par défaut, le texte est activé (`--use_text`). Vous pouvez comparer rapidement :

```bash
python run_research.py --mode research --n_trials 20 --no_text
```

### 4) Lancer l’étude d’ablation

Compare explicitement :

- **Baseline** = variables numériques uniquement
- **Hybrid** = numériques + TF‑IDF

```bash
python run_ablation_study.py --n_trials 10
```

Sorties :

- `research_results/ablation_study/ablation_metrics.csv`
- `research_results/ablation_study/ablation_f0.5.png`
- `research_results/ablation_study/ablation_precision.png`
- `research_results/ablation_study/ablation_recall.png`

## Modèles entraînés

Le pipeline inclut notamment :

- Logistic Regression
- Random Forest
- AdaBoost
- SVM
- MLP
- (et un ensemble stacking selon la configuration)

L’optimisation hyperparamétrique est gérée par **Optuna**.

## Conseils pratiques

- Commencez avec peu d’essais (`--n_trials 5` ou `10`) pour valider le pipeline.
- Montez ensuite à `30+` pour un tuning plus robuste.
- Utilisez `--no_text` pour mesurer le gain réel des features sémantiques sur votre split courant.

## Licence

Ajoutez ici la licence de votre choix (MIT, Apache-2.0, etc.) si nécessaire.
