# Loan Prediction Notebook Refactor — Design Spec
**Date:** 2026-03-14
**Status:** Approved

---

## Overview

Refactor the existing Google Colab-specific Jupyter notebook (`Loan_Prediction_Notebook_ipynb.ipynb`) into a clean, portable, environment-agnostic notebook suitable for a public GitHub portfolio. The notebook must run without modification on local Jupyter, VS Code, and Kaggle Notebooks.

---

## Goals

- Remove all Google Colab dependencies
- Make paths and configuration portable
- Eliminate redundant code
- Reorganize cells into a logical, readable narrative
- Add documentation (docstrings, inline comments, README, requirements.txt)
- Keep the notebook format (no conversion to `.py` scripts)

---

## File Structure

```
Loan-Prediction-XGBoost/
├── Loan_Prediction_Notebook.ipynb   # renamed: remove _ipynb suffix
├── train.csv
├── test_no_label.csv
├── requirements.txt                 # pinned dependencies (captured via pip freeze)
├── .gitignore                       # ignore checkpoints, caches
└── README.md                        # problem, setup, run instructions
```

Data files remain in the same directory as the notebook. No subdirectories needed.

---

## Notebook Structure

The notebook is reorganized into 7 sections with clear markdown headers:

### 1. Setup & Configuration
- `import os` included in imports
- `from sklearn.ensemble import RandomForestClassifier` included in imports
- All other library imports (pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn) — no `!pip install`
- Single CONFIG cell with named constants:
  - `DATA_DIR = '.'`
  - `RANDOM_STATE = 42`
  - `CV_FOLDS = 5`
- `CV_FOLDS` is used as follows:
  - In `evaluate_with_cv()`: the function signature is changed from `(model, X, y, cv_splits)` (accepting a pre-built iterator) to `(model, X, y, cv_folds)` (accepting an integer). The function body is updated to instantiate `StratifiedKFold(n_splits=cv_folds)` internally. Call sites pass `CV_FOLDS` as the integer argument.
  - In `optimize_xgboost()` and `optimize_random_forest()`: already accept `cv_folds` as an integer; pass `CV_FOLDS` at call sites.
- Markdown note listing all required dependencies and pointing to `requirements.txt`

### 2. Data Loading & Exploration
- Load train data once: `train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))`
- Load test data once: `test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_no_label.csv'))`
- These variables are reused in later sections — no duplicate loads anywhere
- Shape, dtypes, missing value summary
- Distribution plots, histograms, boxplots for train data only

### 3. Preprocessing & Feature Engineering
- `preprocess_loan_data(df)` function with full docstring
- Inline comments explaining:
  - Why log transforms are applied to skewed income/loan features
  - What each engineered risk feature captures (e.g. `high_risk_profile`, `financial_stability`)
- One execution cell that calls `preprocess_loan_data()` on a copy of `train_df` and prints feature count before/after (illustrative only — the actual training pipeline in Section 4 re-applies preprocessing to the full `train_df`)

### 4. Model Training
- Helper functions defined first, execution cells after:
  - `evaluate_with_cv(model, X, y, cv_folds)` — cross-validation metrics; docstring; uses `CV_FOLDS`
  - `print_cv_results(results, model_name="Model")` — formatted output helper; docstring; `model_name` parameter is kept
  - `optimize_xgboost(X, y, random_state, cv_folds)` — Phase 1 GridSearchCV; docstring; uses `CV_FOLDS`
  - `optimize_random_forest(X, y, random_state, cv_folds)` — Phase 2 GridSearchCV; docstring; uses `CV_FOLDS`; preceded by a markdown cell explaining it is available but not called by default, with a usage example
- Execution cells (XGBoost only):
  - Preprocess `train_df`, split into `X` and `y`
  - Phase 1: broad `optimize_xgboost()` call
  - Phase 2: fine-tuning micro-optimization around best Phase 1 params

### 5. Evaluation
- CV results display via `print_cv_results()`
- Confusion matrix: new cell added, uses `cross_val_predict(final_best_model, X, y, cv=CV_FOLDS)` to generate predictions for the confusion matrix plot
- Feature importance plot (top 10 features from `final_best_model.feature_importances_`)

### 6. Prediction & Submission
- Reuse `test_df` loaded in Section 2 — no second `pd.read_csv` call
- Apply `preprocess_loan_data()` to `test_df`
- Generate predictions with `final_best_model`
- Build output path: `submission_filename = os.path.join(DATA_DIR, f'submission_{model_source.lower()}_final.csv')` — replaces old `DATA_PATH` string concatenation
- Save submission CSV

### 7. Interpretability (SHAP)
- Three-function structure from the source is kept (not collapsed into one):
  - `analyze_model_interpretability(model, X)` — computes SHAP values; docstring
  - `create_shap_visualizations(shap_values, X)` — summary plot + bar plot; docstring
  - `run_interpretability_analysis(model, X)` — orchestrates the above two; docstring; **the source version takes no arguments and resolves `model`/`X` from globals — this must be changed to accept explicit `(model, X)` parameters and the function body updated accordingly**
- One execution cell calling `run_interpretability_analysis(final_best_model, X)`
- Inline comments explaining what SHAP values represent and how to read the plots

---

## Code Changes

### Remove
- `from google.colab import drive` and `drive.mount('/content/drive')`
- `!pip install shap`
- `import lightgbm as lgb` and all lgb references
- Hardcoded paths (`/content/drive/MyDrive/Eafit/`) everywhere they appear, including the `submission_filename` assignment in the prediction cell
- Duplicate `pd.read_csv` call for train data in the training pipeline cell (reuse `train_df` from Section 2)
- Duplicate `pd.read_csv` call for test data in the prediction cell (reuse `test_df` from Section 2)
- Cell 24 stub (`### CODE HERE ###`) — this placeholder is deleted; the SHAP execution call moves to Section 7

### Replace
- All file path construction → `os.path.join(DATA_DIR, filename)` derived from CONFIG cell
- `DATA_PATH + f'submission_...'` → `os.path.join(DATA_DIR, f'submission_...')`
- Scattered magic numbers (`n_splits=5`, `cv_folds=5`, `random_state=42`) → named constants `CV_FOLDS` and `RANDOM_STATE` from CONFIG cell, passed as arguments
- `run_interpretability_analysis()` (zero-argument, reads globals) → `run_interpretability_analysis(model, X)` (explicit parameters; function body updated to use them instead of global lookups)

### Add
- `import os` to imports
- `from sklearn.ensemble import RandomForestClassifier` to imports
- Docstrings to all 6 functions (`preprocess_loan_data`, `evaluate_with_cv`, `print_cv_results`, `optimize_xgboost`, `optimize_random_forest`, `analyze_model_interpretability`, `create_shap_visualizations`, `run_interpretability_analysis`)
- Inline comments on feature engineering and SHAP interpretation
- Confusion matrix plot cell in Section 5
- Markdown cell before `optimize_random_forest()` with usage example
- `requirements.txt`: pinned versions captured from the development environment via `pip freeze`, filtered to the libraries actually used
- `.gitignore` covering `.ipynb_checkpoints/`, `__pycache__/`, `*.pyc` — submission CSVs are NOT gitignored so portfolio viewers can see example output
- `README.md` with:
  - Problem description (binary loan approval classification, Kaggle-style)
  - Dataset description (features list, target variable `Loan_Status`)
  - Setup instructions (`pip install -r requirements.txt`)
  - How to run (open notebook, run all cells)
  - Approach summary (feature engineering, XGBoost GridSearchCV, SHAP interpretability)
  - Performance note (CV accuracy ~80%+, consistent with notebook baseline goal)

---

## Constraints

- Notebook format only — no conversion to `.py` scripts
- Data files stay in the same directory as the notebook
- Both XGBoost optimization phases are kept
- Random Forest function is kept but not executed
- LightGBM is removed entirely
- Target audience: GitHub portfolio viewers and potential forkers

---

## Out of Scope

- Converting to Python scripts or a package structure
- Adding a `data/` subfolder
- Adding a `FAST_MODE` flag for skipping Phase 2
- Building out the LightGBM model
- Adding new models or features beyond the existing pipeline
