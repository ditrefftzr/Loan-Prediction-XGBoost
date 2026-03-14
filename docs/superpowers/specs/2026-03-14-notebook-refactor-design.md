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
├── requirements.txt                 # pinned dependencies
├── .gitignore                       # ignore checkpoints, caches, submissions
└── README.md                        # problem, setup, run instructions
```

Data files remain in the same directory as the notebook. No subdirectories needed.

---

## Notebook Structure

The notebook is reorganized into 7 sections with clear markdown headers:

### 1. Setup & Configuration
- All library imports (no `!pip install`)
- Single CONFIG cell with named constants:
  - `DATA_DIR = '.'`
  - `RANDOM_STATE = 42`
  - `CV_FOLDS = 5`
- Markdown note listing dependencies and pointing to `requirements.txt`

### 2. Data Loading & Exploration
- Load train/test using `os.path.join(DATA_DIR, 'train.csv')` and `os.path.join(DATA_DIR, 'test_no_label.csv')`
- Shape, dtypes, missing value summary
- Distribution plots, histograms, boxplots
- Single load of each file (no duplicate `pd.read_csv` calls)

### 3. Preprocessing & Feature Engineering
- `preprocess_loan_data(df)` function with full docstring
- Inline comments on non-obvious logic:
  - Why log transforms are applied
  - What each engineered risk feature captures
- One execution cell showing feature count before/after preprocessing

### 4. Model Training
- Helper functions (grouped before execution cells):
  - `evaluate_model()` — cross-validation metrics with docstring
  - `print_cv_results()` — formatted output helper with docstring
  - `optimize_xgboost()` — GridSearchCV Phase 1 with docstring
  - `optimize_random_forest()` — GridSearchCV function with docstring; markdown note explaining it is available but not executed by default, with instructions on how to call it
- Execution cells:
  - Phase 1: broad XGBoost GridSearchCV
  - Phase 2: fine-tuning micro-optimization

### 5. Evaluation
- CV results display
- Confusion matrix
- Feature importance plot (top 10)

### 6. Prediction & Submission
- Load test data, apply `preprocess_loan_data()`
- Generate predictions with best model
- Save `submission_{model}_final.csv` to `DATA_DIR`

### 7. Interpretability (SHAP)
- `shap_analysis()` function with docstring
- Execution cell
- SHAP summary plot and bar plot visualizations

---

## Code Changes

### Remove
- `from google.colab import drive` and `drive.mount('/content/drive')`
- `!pip install shap`
- `import lightgbm as lgb` and all lgb references
- Hardcoded paths (`/content/drive/MyDrive/Eafit/`)
- Duplicate `pd.read_csv` calls for train data (loaded once in Section 2, reused in Section 4)

### Replace
- All file paths → `os.path.join(DATA_DIR, filename)` derived from CONFIG cell
- Scattered magic numbers → named constants in CONFIG cell

### Add
- Docstrings to all 4 main functions
- Inline comments explaining feature engineering decisions and SHAP interpretation
- Markdown cell before `optimize_random_forest()` explaining it is available but not called, with usage example
- `requirements.txt` with pinned library versions
- `.gitignore` covering `.ipynb_checkpoints/`, `__pycache__/`, `*.pyc`, `submission_*.csv`
- `README.md` with:
  - Problem description (binary loan approval classification)
  - Dataset description (features, target variable)
  - Setup instructions (`pip install -r requirements.txt`)
  - How to run the notebook
  - Brief explanation of the approach (feature engineering + XGBoost + SHAP)

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
