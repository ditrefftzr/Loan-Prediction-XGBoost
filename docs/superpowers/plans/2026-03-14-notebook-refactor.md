# Loan Prediction Notebook Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform `Loan_Prediction_Notebook_ipynb.ipynb` from a Google Colab-specific notebook into a clean, portable, portfolio-ready notebook that runs on local Jupyter, VS Code, and Kaggle without modification.

**Architecture:** Build a new notebook file (`Loan_Prediction_Notebook.ipynb`) with 7 clearly labeled sections by reading the existing notebook's cell content and writing a fully restructured version. Supporting files (requirements.txt, .gitignore, README.md) are added alongside. The old notebook is removed via `git rm`.

**Tech Stack:** Python 3, pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn

---

## Implementation Note: How to Build the New Notebook

The new notebook is built using a Python script that:
1. Reads the old notebook JSON to extract reusable cell source
2. Constructs new cells as Python strings
3. Assembles a valid `.ipynb` JSON structure
4. Writes `Loan_Prediction_Notebook.ipynb`

A valid `.ipynb` cell has the form:
```python
{
  "cell_type": "code" | "markdown",
  "metadata": {},
  "source": ["line1\n", "line2\n"],  # list of strings
  "outputs": [],                      # code cells only
  "execution_count": None            # code cells only
}
```
Notebook kernel info: `{"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.x.x"}}`

---

## Chunk 1: Supporting Files

### Task 1: Create .gitignore

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Write .gitignore**

```
.ipynb_checkpoints/
__pycache__/
*.pyc
```

- [ ] **Step 2: Verify**

Run: `cat .gitignore`
Expected: three lines above

---

### Task 2: Create requirements.txt

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Check versions installed**

Run: `pip show pandas numpy scikit-learn xgboost shap matplotlib seaborn`
Record the version for each library.

- [ ] **Step 2: Write requirements.txt using the versions captured above**

Format (replace X.X.X with actual versions):
```
pandas==X.X.X
numpy==X.X.X
scikit-learn==X.X.X
xgboost==X.X.X
shap==X.X.X
matplotlib==X.X.X
seaborn==X.X.X
```

- [ ] **Step 3: Verify**

Run: `cat requirements.txt`
Expected: 7 lines with pinned versions.

---

### Task 3: Create README.md

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README.md with this exact content**

```markdown
# Loan Prediction with XGBoost

Binary loan approval classification using XGBoost with GridSearchCV optimization and SHAP interpretability.

## Problem Description

Predict whether a loan application will be approved (binary classification: 0 = Rejected, 1 = Approved). Based on a Kaggle-style competition dataset.

## Dataset

Features: Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area

Target: `Loan_Status` (0 or 1)

- `train.csv` — labeled training data
- `test_no_label.csv` — test data without labels

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

Open `Loan_Prediction_Notebook.ipynb` in Jupyter, VS Code, or Kaggle Notebooks and run all cells in order.

No path changes needed — data files are expected in the same directory as the notebook.

## Approach

1. **Feature Engineering** — log transforms on skewed income/loan features, engineered risk features (`high_risk_profile`, `financial_stability`, `wealthy_conservative_borrower`)
2. **XGBoost GridSearchCV** — Phase 1 broad search (2,187 combinations), Phase 2 fine-tuning around best params
3. **SHAP Interpretability** — global feature importance and per-prediction explanations

## Performance

Cross-validated accuracy: ~80%+ (5-fold StratifiedKFold, F1-optimized)
```

- [ ] **Step 2: Commit supporting files**

```bash
git add requirements.txt .gitignore README.md
git commit -m "chore: add requirements.txt, .gitignore, and README"
```

---

## Chunk 2: Notebook — Sections 1 & 2

### Task 4: Build new notebook with Sections 1 and 2

**Files:**
- Create: `Loan_Prediction_Notebook.ipynb`

- [ ] **Step 1: Read original cell 0 (intro markdown)**

Read `Loan_Prediction_Notebook_ipynb.ipynb` and extract cell 0 source. Remove any Google Colab-specific links (Google Drive links). Keep the general description text.

- [ ] **Step 2: Write a Python builder script and run it to create the initial notebook**

Create a temporary file `build_notebook.py` with this content, then run it:

```python
import json

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else source.splitlines(keepends=True)
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else source.splitlines(keepends=True)
    }

cells = []

# --- Cell 0: Intro markdown (from original cell 0, stripped of Colab links) ---
cells.append(md_cell(
    "# Loan Prediction Competition\n\n"
    "In this notebook, we apply ensemble methods such as XGBoost for binary loan approval classification, "
    "featuring systematic hyperparameter optimization and SHAP model interpretability.\n"
))

# --- Cell 1: Section 1 header ---
cells.append(md_cell("## 1. Setup & Configuration\n"))

# --- Cell 2: Imports ---
cells.append(code_cell(
    "import os\n"
    "import shap\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "from sklearn.preprocessing import LabelEncoder\n"
    "from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV\n"
    "from sklearn.metrics import (\n"
    "    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,\n"
    "    classification_report, confusion_matrix, ConfusionMatrixDisplay\n"
    ")\n"
    "from sklearn.ensemble import RandomForestClassifier\n"
    "from xgboost import XGBClassifier\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
))

# --- Cell 3: CONFIG ---
cells.append(code_cell(
    "# Configuration constants — edit DATA_DIR to point to a different directory\n"
    "DATA_DIR = '.'\n"
    "RANDOM_STATE = 42\n"
    "CV_FOLDS = 5\n"
))

# --- Cell 4: Dependencies note ---
cells.append(md_cell(
    "**Dependencies:** pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn\n\n"
    "Install with: `pip install -r requirements.txt`\n"
))

# --- Cell 5: Section 2 header ---
cells.append(md_cell(
    "## 2. Data Loading & Exploration\n\n"
    "Train and test data are loaded once here and reused in later sections — no duplicate loads.\n"
))

# --- Cell 6: Load data ---
cells.append(code_cell(
    "train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))\n"
    "test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_no_label.csv'))\n"
    "print(f'Train shape: {train_df.shape}')\n"
    "print(f'Test shape: {test_df.shape}')\n"
    "train_df.head()\n"
))

# --- Cell 7: Describe ---
cells.append(code_cell(
    "train_df.describe()\n"
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

with open('Loan_Prediction_Notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Created notebook with {len(cells)} cells")
```

Run: `python build_notebook.py`
Expected: `Created notebook with 8 cells`

- [ ] **Step 3: Read original cells 7 and 8 from the old notebook**

Run:
```python
import json
nb_old = json.load(open('Loan_Prediction_Notebook_ipynb.ipynb', encoding='utf-8'))
print('Cell 7:', ''.join(nb_old['cells'][7]['source'])[:200])
print('Cell 8:', ''.join(nb_old['cells'][8]['source'])[:200])
```

Note the exact content — you will add these cells in the next step.

- [ ] **Step 4: Append distribution and missing-value cells to the notebook**

Load the saved notebook, append two cells, and save:

```python
import json

with open('Loan_Prediction_Notebook.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
with open('Loan_Prediction_Notebook_ipynb.ipynb', encoding='utf-8') as f:
    nb_old = json.load(f)

def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}

# Cell 8 (distribution plots): use old cell 7 source, replacing 'data' with 'train_df'
old_cell7_src = ''.join(nb_old['cells'][7]['source']).replace('data[', 'train_df[').replace('data.', 'train_df.')
nb['cells'].append(code_cell(old_cell7_src))

# Cell 9 (missing values): use old cell 8 source, replacing 'data' with 'train_df'
old_cell8_src = ''.join(nb_old['cells'][8]['source']).replace('data[', 'train_df[').replace('data.', 'train_df.')
nb['cells'].append(code_cell(old_cell8_src))

with open('Loan_Prediction_Notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook now has {len(nb['cells'])} cells")
```

Expected: `Notebook now has 10 cells`

- [ ] **Step 5: Verify JSON is valid**

Run: `python -c "import json; nb=json.load(open('Loan_Prediction_Notebook.ipynb', encoding='utf-8')); print(f'Cells: {len(nb[\"cells\"])}')"`
Expected: `Cells: 10`

---

## Chunk 3: Notebook — Section 3

### Task 5: Add Section 3 — Preprocessing & Feature Engineering

**Files:**
- Modify: `Loan_Prediction_Notebook.ipynb`

- [ ] **Step 1: Read original cell 10 source (preprocess_loan_data)**

Run:
```python
import json
nb_old = json.load(open('Loan_Prediction_Notebook_ipynb.ipynb', encoding='utf-8'))
src = ''.join(nb_old['cells'][10]['source'])
print(src)
```

- [ ] **Step 2: Append Section 3 cells to the notebook**

```python
import json

with open('Loan_Prediction_Notebook.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
with open('Loan_Prediction_Notebook_ipynb.ipynb', encoding='utf-8') as f:
    nb_old = json.load(f)

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}

def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}

# Cell 10: Section 3 header
nb['cells'].append(md_cell(
    "## 3. Preprocessing & Feature Engineering\n\n"
    "The preprocessing pipeline handles missing values via intelligent imputation, "
    "encodes categorical features, applies log transforms to reduce skew in income "
    "and loan amount columns, and engineers domain-specific risk features.\n"
))

# Cell 11: preprocess_loan_data — read from old cell 10, apply these three changes:
#   1. Replace Spanish docstring with English
#   2. Insert log-transform inline comment before the log transforms block (line ~52)
#   3. Insert risk-feature inline comment before the engineered features block (line ~90)
preprocess_src = ''.join(nb_old['cells'][10]['source'])

# Change 1: Replace docstring
old_docstring = '    """\n    Pipeline de preprocesamiento optimizado para datos de préstamos\n    Versión limpia que logró 80% de accuracy\n\n    Input: DataFrame crudo\n    Output: DataFrame procesado listo para XGBoost\n    """'
new_docstring = (
    '    """\n'
    '    Preprocessing pipeline for loan prediction data.\n\n'
    '    Handles missing values via intelligent imputation, encodes categorical\n'
    '    features, applies log transforms to reduce skew, and engineers\n'
    '    domain-specific risk features.\n\n'
    '    Args:\n'
    '        df: Raw loan DataFrame (train or test set)\n\n'
    '    Returns:\n'
    '        Processed DataFrame ready for model training\n'
    '    """'
)
preprocess_src = preprocess_src.replace(old_docstring, new_docstring, 1)

# Change 2: Add inline comment before log transforms (before the line with '# 2. TRANSFORMACIONES')
log_comment = (
    "    # Log transforms reduce right skew in income and loan amount features.\n"
    "    # Normalized distributions improve XGBoost split quality.\n"
)
preprocess_src = preprocess_src.replace(
    "    # 2. TRANSFORMACIONES LOGAR",
    log_comment + "    # 2. TRANSFORMACIONES LOGAR",
    1
)

# Change 3: Add inline comment before engineered risk features (before high_risk_profile)
risk_comment = (
    "    # Engineered risk features capture domain knowledge about loan creditworthiness:\n"
    "    # - high_risk_profile: applicant has multiple simultaneous risk factors\n"
    "    # - financial_stability: stable married income, no outstanding debt\n"
)
preprocess_src = preprocess_src.replace(
    "    data['high_risk_profile']",
    risk_comment + "    data['high_risk_profile']",
    1
)

nb['cells'].append(code_cell(preprocess_src))

# Cell 12: Illustrative execution
nb['cells'].append(code_cell(
    "# Illustrative: show feature count before and after preprocessing.\n"
    "# The actual training pipeline in Section 4 re-applies this to train_df.\n"
    "_processed = preprocess_loan_data(train_df.copy())\n"
    "print(f'Features before preprocessing: {train_df.shape[1]}')\n"
    "print(f'Features after preprocessing: {_processed.shape[1]}')\n"
))

with open('Loan_Prediction_Notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook now has {len(nb['cells'])} cells")
```

Expected: `Notebook now has 13 cells`

- [ ] **Step 3: Verify JSON is valid**

Run: `python -c "import json; nb=json.load(open('Loan_Prediction_Notebook.ipynb', encoding='utf-8')); print(f'Cells: {len(nb[\"cells\"])}')"`
Expected: `Cells: 13`

---

## Chunk 4: Notebook — Section 4

### Task 6: Add Section 4 — Model Training

**Files:**
- Modify: `Loan_Prediction_Notebook.ipynb`

This task adds all Section 4 cells: helper function definitions, the Random Forest markdown note, and execution cells for Phase 1 and Phase 2.

- [ ] **Step 1: Append Section 4 cells — header, helper functions, optimizer definitions**

```python
import json

with open('Loan_Prediction_Notebook.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
with open('Loan_Prediction_Notebook_ipynb.ipynb', encoding='utf-8') as f:
    nb_old = json.load(f)

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}

def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}

# Cell 13: Section 4 header
nb['cells'].append(md_cell(
    "## 4. Model Training\n\n"
    "### 4.1 Helper Functions\n"
))

# Cell 14: evaluate_with_cv + print_cv_results
# Signature changed: cv_splits (iterator) → cv_folds (integer)
# StratifiedKFold is now instantiated inside the function
# quick_evaluate is removed
nb['cells'].append(code_cell(
    "def evaluate_with_cv(model, X, y, cv_folds):\n"
    '    """\n'
    "    Evaluate a model using stratified k-fold cross-validation.\n\n"
    "    Args:\n"
    "        model: Scikit-learn compatible classifier\n"
    "        X: Feature DataFrame\n"
    "        y: Target Series\n"
    "        cv_folds: Number of CV folds (integer)\n\n"
    "    Returns:\n"
    "        dict with mean metrics ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')\n"
    "        and 'std' sub-dict with per-metric standard deviations\n"
    '    """\n'
    "    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)\n"
    "    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}\n\n"
    "    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):\n"
    "        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]\n"
    "        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]\n\n"
    "        model.fit(X_train, y_train)\n"
    "        y_pred = model.predict(X_val)\n"
    "        y_pred_proba = model.predict_proba(X_val)[:, 1]\n\n"
    "        metrics['accuracy'].append(accuracy_score(y_val, y_pred))\n"
    "        metrics['precision'].append(precision_score(y_val, y_pred, pos_label=1))\n"
    "        metrics['recall'].append(recall_score(y_val, y_pred, pos_label=1))\n"
    "        metrics['f1'].append(f1_score(y_val, y_pred, pos_label=1))\n"
    "        metrics['roc_auc'].append(roc_auc_score(y_val == 1, y_pred_proba))\n\n"
    "    results = {metric: np.mean(values) for metric, values in metrics.items()}\n"
    "    results['std'] = {metric: np.std(values) for metric, values in metrics.items()}\n"
    "    return results\n\n\n"
    "def print_cv_results(results, model_name='Model'):\n"
    '    """\n'
    "    Print cross-validation results in a readable format.\n\n"
    "    Args:\n"
    "        results: Output dict from evaluate_with_cv()\n"
    "        model_name: Label for the model in output headers\n"
    '    """\n'
    "    print(f'\\n{model_name} - CV Results:')\n"
    "    print(f\"  Accuracy:  {results['accuracy']:.4f} (\\u00b1{results['std']['accuracy']:.4f})\")\n"
    "    print(f\"  F1-Score:  {results['f1']:.4f} (\\u00b1{results['std']['f1']:.4f})\")\n"
    "    print(f\"  Precision: {results['precision']:.4f} (\\u00b1{results['std']['precision']:.4f})\")\n"
    "    print(f\"  Recall:    {results['recall']:.4f} (\\u00b1{results['std']['recall']:.4f})\")\n"
    "    print(f\"  ROC-AUC:   {results['roc_auc']:.4f} (\\u00b1{results['std']['roc_auc']:.4f})\")\n"
))

# Cell 15: optimize_xgboost — add random_state, cv_folds as explicit params; add English docstring
nb['cells'].append(code_cell(
    "def optimize_xgboost(X, y, random_state, cv_folds):\n"
    '    """\n'
    "    Optimize XGBoost hyperparameters with GridSearchCV (Phase 1: broad search).\n\n"
    "    Grid is centered around known high-performing parameters (~80% baseline).\n"
    "    Total combinations: 3^7 = 2,187.\n\n"
    "    Args:\n"
    "        X: Feature DataFrame\n"
    "        y: Target Series\n"
    "        random_state: Random seed for reproducibility\n"
    "        cv_folds: Number of CV folds (integer)\n\n"
    "    Returns:\n"
    "        dict with 'model', 'score', 'params', 'improvement'\n"
    '    """\n'
    "    param_grid = {\n"
    "        'learning_rate': [0.02, 0.03, 0.04],\n"
    "        'n_estimators': [175, 200, 225],\n"
    "        'max_depth': [3, 4, 5],\n"
    "        'min_child_weight': [2, 3, 4],\n"
    "        'subsample': [0.92, 0.95, 0.98],\n"
    "        'colsample_bytree': [0.72, 0.75, 0.78],\n"
    "        'scale_pos_weight': [2.25, 2.5, 2.75]\n"
    "    }\n\n"
    "    xgb = XGBClassifier(\n"
    "        random_state=random_state,\n"
    "        eval_metric='logloss',\n"
    "        n_jobs=-1,\n"
    "        verbosity=0\n"
    "    )\n"
    "    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)\n"
    "    grid_search = GridSearchCV(\n"
    "        estimator=xgb, param_grid=param_grid, cv=cv,\n"
    "        scoring='f1', n_jobs=-1, verbose=1\n"
    "    )\n\n"
    "    print('Optimizing XGBoost (Phase 1)...')\n"
    "    print(f'  Combinations to try: {3**7}')\n"
    "    print('  Estimated time: ~15-20 minutes')\n\n"
    "    grid_search.fit(X, y)\n\n"
    "    print(f'Best F1-Score: {grid_search.best_score_:.4f}')\n"
    "    print(f'Best parameters: {grid_search.best_params_}')\n\n"
    "    return {\n"
    "        'model': grid_search.best_estimator_,\n"
    "        'score': grid_search.best_score_,\n"
    "        'params': grid_search.best_params_,\n"
    "        'improvement': grid_search.best_score_ > 0.80\n"
    "    }\n"
))

# Cell 16: Random Forest markdown note
nb['cells'].append(md_cell(
    "### Random Forest (Optional)\n\n"
    "`optimize_random_forest()` is defined below but **not called by default**. "
    "It is available for comparison experiments.\n\n"
    "Usage example:\n"
    "```python\n"
    "rf_results = optimize_random_forest(X, y, RANDOM_STATE, CV_FOLDS)\n"
    "print_cv_results(evaluate_with_cv(rf_results['model'], X, y, CV_FOLDS), 'Random Forest')\n"
    "```\n"
))

# Cell 17: optimize_random_forest — add random_state, cv_folds as explicit params; English docstring
# Keep only optimize_random_forest from original cell 16 (remove optimize_gradient_boosting, compare_multiple_models)
nb['cells'].append(code_cell(
    "def optimize_random_forest(X, y, random_state, cv_folds):\n"
    '    """\n'
    "    Optimize RandomForest hyperparameters with GridSearchCV.\n\n"
    "    Not called by default — see the markdown cell above for usage.\n"
    "    Total combinations: 3*3*3*3*1*1 = 81.\n\n"
    "    Args:\n"
    "        X: Feature DataFrame\n"
    "        y: Target Series\n"
    "        random_state: Random seed for reproducibility\n"
    "        cv_folds: Number of CV folds (integer)\n\n"
    "    Returns:\n"
    "        dict with 'model', 'score', 'params'\n"
    '    """\n'
    "    param_grid = {\n"
    "        'n_estimators': [100, 200, 300],\n"
    "        'max_depth': [5, 10, None],\n"
    "        'min_samples_split': [2, 5, 10],\n"
    "        'min_samples_leaf': [1, 2, 4],\n"
    "        'max_features': ['sqrt'],\n"
    "        'class_weight': ['balanced']\n"
    "    }\n\n"
    "    rf = RandomForestClassifier(random_state=random_state)\n"
    "    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)\n"
    "    grid_search = GridSearchCV(\n"
    "        estimator=rf, param_grid=param_grid, cv=cv,\n"
    "        scoring='f1', n_jobs=-1, verbose=1\n"
    "    )\n\n"
    "    print('Optimizing Random Forest...')\n"
    "    grid_search.fit(X, y)\n\n"
    "    print(f'Best F1-Score: {grid_search.best_score_:.4f}')\n"
    "    print(f'Best parameters: {grid_search.best_params_}')\n\n"
    "    return {\n"
    "        'model': grid_search.best_estimator_,\n"
    "        'score': grid_search.best_score_,\n"
    "        'params': grid_search.best_params_\n"
    "    }\n"
))

with open('Loan_Prediction_Notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook now has {len(nb['cells'])} cells")
```

Expected: `Notebook now has 18 cells`

- [ ] **Step 2: Append Section 4 execution cells (Phase 1 and Phase 2)**

```python
import json

with open('Loan_Prediction_Notebook.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}

def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}

# Cell 18: Section 4.2 header
nb['cells'].append(md_cell("### 4.2 Execution\n"))

# Cell 19: Preprocess train_df + Phase 1 optimize_xgboost
# Replaces old cell 18's duplicate pd.read_csv with train_df reuse
# Replaces cv_splits with CV_FOLDS in evaluate_with_cv call
nb['cells'].append(code_cell(
    "# Preprocess training data and split into features and target\n"
    "data_processed = preprocess_loan_data(train_df)\n"
    "X = (\n"
    "    data_processed.drop(['Loan_Status', 'id'], axis=1)\n"
    "    if 'id' in data_processed.columns\n"
    "    else data_processed.drop('Loan_Status', axis=1)\n"
    ")\n"
    "y = data_processed['Loan_Status']\n"
    "print(f'Training features: {X.shape[1]}, Samples: {X.shape[0]}')\n\n"
    "# Phase 1: broad hyperparameter search\n"
    "xgb_results = optimize_xgboost(X, y, RANDOM_STATE, CV_FOLDS)\n\n"
    "# Detailed evaluation of Phase 1 best model\n"
    "phase1_results = evaluate_with_cv(xgb_results['model'], X, y, CV_FOLDS)\n"
    "print_cv_results(phase1_results, 'XGBoost Phase 1')\n"
))

# Cell 20: Phase 2 fine-tuning
# Replaces n_splits=5 with CV_FOLDS, removes cv_splits variable,
# updates evaluate_with_cv calls to pass CV_FOLDS integer
nb['cells'].append(code_cell(
    "# Phase 2: fine-tuning — micro-optimization around Phase 1 best params\n"
    "print('\\n' + '='*60)\n"
    "print('PHASE 2: FINE-TUNING')\n"
    "print('='*60)\n\n"
    "base_params = xgb_results['params']\n"
    "print(f'Base parameters: {base_params}')\n\n"
    "# Build fine-grained grid around Phase 1 optimal params\n"
    "param_grid = {}\n\n"
    "base_lr = base_params.get('learning_rate', 0.03)\n"
    "param_grid['learning_rate'] = [max(0.01, base_lr - 0.005), base_lr, min(0.1, base_lr + 0.005)]\n\n"
    "base_n_est = base_params.get('n_estimators', 200)\n"
    "param_grid['n_estimators'] = [max(50, base_n_est - 25), base_n_est, base_n_est + 25]\n\n"
    "base_colsample = base_params.get('colsample_bytree', 0.75)\n"
    "param_grid['colsample_bytree'] = [\n"
    "    max(0.5, base_colsample - 0.05), base_colsample, min(1.0, base_colsample + 0.05)\n"
    "]\n\n"
    "base_scale = base_params.get('scale_pos_weight', 2.5)\n"
    "param_grid['scale_pos_weight'] = [max(1.0, base_scale - 0.25), base_scale, base_scale + 0.25]\n\n"
    "# Keep remaining params fixed at Phase 1 best values\n"
    "param_grid['max_depth'] = [base_params.get('max_depth', 3)]\n"
    "param_grid['min_child_weight'] = [base_params.get('min_child_weight', 3)]\n"
    "param_grid['subsample'] = [base_params.get('subsample', 0.95)]\n\n"
    "xgb_fine = XGBClassifier(\n"
    "    random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1, verbosity=0\n"
    ")\n"
    "cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)\n"
    "fine_grid_search = GridSearchCV(\n"
    "    estimator=xgb_fine, param_grid=param_grid, cv=cv,\n"
    "    scoring='f1', n_jobs=-1, verbose=1\n"
    ")\n"
    "fine_grid_search.fit(X, y)\n\n"
    "fine_tune_results = {\n"
    "    'model': fine_grid_search.best_estimator_,\n"
    "    'params': fine_grid_search.best_params_,\n"
    "    'score': fine_grid_search.best_score_\n"
    "}\n"
    "fine_detailed = evaluate_with_cv(fine_tune_results['model'], X, y, CV_FOLDS)\n"
    "print_cv_results(fine_detailed, 'XGBoost Phase 2 (Fine-tuned)')\n\n"
    "# Select the best model between Phase 1 and Phase 2\n"
    "if fine_detailed['f1'] > phase1_results['f1']:\n"
    "    print('\\nDecision: Using FINE-TUNED model (higher F1)')\n"
    "    final_best_model = fine_tune_results['model']\n"
    "    final_best_params = fine_tune_results['params']\n"
    "    final_best_results = fine_detailed\n"
    "    model_source = 'Fine-tuned'\n"
    "else:\n"
    "    print('\\nDecision: Keeping PHASE 1 model (fine-tuning did not improve)')\n"
    "    final_best_model = xgb_results['model']\n"
    "    final_best_params = xgb_results['params']\n"
    "    final_best_results = phase1_results\n"
    "    model_source = 'Optimized'\n"
))

with open('Loan_Prediction_Notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook now has {len(nb['cells'])} cells")
```

Expected: `Notebook now has 21 cells`

- [ ] **Step 3: Verify JSON is valid and cell count is correct**

Run: `python -c "import json; nb=json.load(open('Loan_Prediction_Notebook.ipynb', encoding='utf-8')); print(f'Cells: {len(nb[\"cells\"])}')"`
Expected: `Cells: 21`

---

## Chunk 5: Notebook — Sections 5, 6, 7

### Task 7: Add Sections 5, 6, and 7

**Files:**
- Modify: `Loan_Prediction_Notebook.ipynb`

- [ ] **Step 1: Append Sections 5 and 6 cells**

```python
import json

with open('Loan_Prediction_Notebook.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}

def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}

# --- SECTION 5 ---

# Cell 21: Section 5 header
nb['cells'].append(md_cell("## 5. Evaluation\n"))

# Cell 22: CV results display
nb['cells'].append(code_cell(
    "print_cv_results(final_best_results, f'XGBoost {model_source}')\n"
))

# Cell 23: Confusion matrix (NEW)
nb['cells'].append(code_cell(
    "# Confusion matrix using cross-validated predictions\n"
    "y_pred_cv = cross_val_predict(final_best_model, X, y, cv=CV_FOLDS)\n"
    "ConfusionMatrixDisplay.from_predictions(\n"
    "    y, y_pred_cv, display_labels=['Rejected', 'Approved']\n"
    ")\n"
    f"plt.title(f'Confusion Matrix ({CV_FOLDS}-Fold Cross-Validated Predictions)')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# Cell 24: Feature importance plot
nb['cells'].append(code_cell(
    "feature_importance = (\n"
    "    pd.DataFrame({\n"
    "        'feature': X.columns,\n"
    "        'importance': final_best_model.feature_importances_\n"
    "    })\n"
    "    .sort_values('importance', ascending=False)\n"
    "    .head(10)\n"
    ")\n\n"
    "plt.figure(figsize=(10, 6))\n"
    "plt.barh(feature_importance['feature'][::-1], feature_importance['importance'][::-1])\n"
    "plt.xlabel('Feature Importance')\n"
    "plt.title('Top 10 Feature Importances (XGBoost)')\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# --- SECTION 6 ---

# Cell 25: Section 6 header
nb['cells'].append(md_cell(
    "## 6. Prediction & Submission\n\n"
    "Uses `test_df` loaded in Section 2 — no duplicate file read.\n"
))

# Cell 26: Prediction and submission
nb['cells'].append(code_cell(
    "# Reuse test_df from Section 2\n"
    "print(f'Test dataset shape: {test_df.shape}')\n"
    "test_ids = test_df['id'].copy()\n\n"
    "# Apply the same preprocessing pipeline used in training\n"
    "test_processed = preprocess_loan_data(test_df)\n"
    "X_test = test_processed.copy()\n"
    "if 'id' in X_test.columns:\n"
    "    X_test = X_test.drop('id', axis=1)\n\n"
    "# Align test columns with training feature set\n"
    "missing_cols = set(X.columns) - set(X_test.columns)\n"
    "extra_cols = set(X_test.columns) - set(X.columns)\n"
    "for col in missing_cols:\n"
    "    X_test[col] = 0\n"
    "X_test = X_test.drop(columns=extra_cols)\n"
    "X_test = X_test.reindex(columns=X.columns, fill_value=0)\n"
    "print(f'Test features aligned: {X_test.shape[1]}')\n\n"
    "# Generate predictions with the best model\n"
    "test_predictions = final_best_model.predict(X_test)\n\n"
    "# Build submission DataFrame\n"
    "submission = pd.DataFrame({\n"
    "    'id': test_ids,\n"
    "    'pred': test_predictions.astype(int)\n"
    "})\n\n"
    "# Save to file using DATA_DIR config constant\n"
    "submission_filename = os.path.join(DATA_DIR, f'submission_{model_source.lower()}_final.csv')\n"
    "submission.to_csv(submission_filename, index=False)\n"
    "print(f'Submission saved: {submission_filename}')\n"
    "print(submission.head(10).to_string(index=False))\n"
))

with open('Loan_Prediction_Notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook now has {len(nb['cells'])} cells")
```

Expected: `Notebook now has 27 cells`

- [ ] **Step 2: Append Section 7 cells**

```python
import json

with open('Loan_Prediction_Notebook.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
with open('Loan_Prediction_Notebook_ipynb.ipynb', encoding='utf-8') as f:
    nb_old = json.load(f)

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}

def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(keepends=True)}

# Cell 27: Section 7 header
nb['cells'].append(md_cell(
    "## 7. Interpretability (SHAP)\n\n"
    "SHAP (SHapley Additive exPlanations) values measure each feature's contribution "
    "to a specific prediction. Positive SHAP values push the prediction toward loan "
    "approval; negative values push toward rejection.\n"
))

# Cell 28: SHAP functions
# Read original cell 25 (has analyze_model_interpretability, create_shap_visualizations,
# and run_interpretability_analysis). Keep the first two functions unchanged.
# Replace run_interpretability_analysis to accept explicit (model, X) params
# instead of reading from globals.

shap_src = ''.join(nb_old['cells'][25]['source'])

# Replace the run_interpretability_analysis function definition
old_run_fn = (
    "def run_interpretability_analysis():\n"
    '    """\n'
    "    Ejecutar análisis completo de interpretabilidad\n"
    "    Usar DESPUÉS de tener el modelo final\n"
    '    """\n'
    "\n"
    "    print(\"INICIANDO ANÁLISIS DE INTERPRETABILIDAD\")\n"
    "    print(\"=\" * 60)\n"
    "\n"
    "    # Determinar qué modelo usar\n"
    "    if 'final_best_model' in locals() or 'final_best_model' in globals():\n"
    "        model_to_analyze = final_best_model\n"
    "        model_name = \"Final Best Model\"\n"
    "    elif 'xgb_results' in locals() or 'xgb_results' in globals():\n"
    "        model_to_analyze = xgb_results['model']\n"
    "        model_name = \"XGBoost Optimized\"\n"
    "    else:\n"
    "        print(\"No se encontró modelo entrenado. Ejecutar pipeline de optimización primero.\")\n"
    "        return None\n"
    "\n"
    "    # Ejecutar análisis\n"
    "    shap_results = analyze_model_interpretability(model_to_analyze, X, y, model_name)\n"
    "\n"
    "    if shap_results:\n"
    "        print(f\"\\nAnálisis completado. Para visualizaciones:\")\n"
    "        print(\"create_shap_visualizations(shap_results)\")\n"
    "\n"
    "    return shap_results"
)

new_run_fn = (
    "def run_interpretability_analysis(model, X, y):\n"
    '    """\n'
    "    Run complete SHAP interpretability analysis.\n\n"
    "    analyze_model_interpretability retains its original signature\n"
    "    (model, X, y, model_name='XGBoost') where model_name has a default.\n\n"
    "    Args:\n"
    "        model: Trained classifier (XGBoost or compatible tree model)\n"
    "        X: Feature DataFrame used for training\n"
    "        y: Target Series (used for error analysis — False Positives/Negatives)\n\n"
    "    Returns:\n"
    "        dict with explainer, shap_values, feature_importance, X_sample\n"
    '    """\n'
    "    print('STARTING INTERPRETABILITY ANALYSIS')\n"
    "    print('=' * 60)\n"
    "\n"
    "    shap_results = analyze_model_interpretability(model, X, y)\n"
    "\n"
    "    if shap_results:\n"
    "        print('\\nAnalysis complete.')\n"
    "        print('To visualize: create_shap_visualizations(shap_results)')\n"
    "\n"
    "    return shap_results"
)

shap_src = shap_src.replace(old_run_fn, new_run_fn, 1)
nb['cells'].append(code_cell(shap_src))

# Cell 29: SHAP execution — all three args explicit, no global dependencies
nb['cells'].append(code_cell(
    "shap_results = run_interpretability_analysis(final_best_model, X, y)\n"
))

# Cell 30: SHAP visualizations
nb['cells'].append(code_cell(
    "create_shap_visualizations(shap_results)\n"
))

with open('Loan_Prediction_Notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook now has {len(nb['cells'])} cells")
```

Expected: `Notebook now has 31 cells`

- [ ] **Step 3: Verify JSON is valid and all 31 cells are present**

Run: `python -c "import json; nb=json.load(open('Loan_Prediction_Notebook.ipynb', encoding='utf-8')); [print(i, nb['cells'][i]['cell_type'], ''.join(nb['cells'][i]['source'])[:60].replace(chr(10),' ')) for i in range(len(nb['cells']))]"`

Confirm:
- Cell 0: markdown intro
- Cell 2: imports (contains `import os`, `RandomForestClassifier`, no `!pip`, no `lightgbm`)
- Cell 3: CONFIG (`DATA_DIR`, `RANDOM_STATE`, `CV_FOLDS`)
- Cell 6: `train_df = pd.read_csv(os.path.join(DATA_DIR,`)
- Cell 14: `def evaluate_with_cv(model, X, y, cv_folds):`
- Cell 19: `data_processed = preprocess_loan_data(train_df)` (no pd.read_csv)
- Cell 26: `submission_filename = os.path.join(DATA_DIR,`
- Cell 28: `def run_interpretability_analysis(model, X, y):`
- Cell 29: `shap_results = run_interpretability_analysis(final_best_model, X, y)`

---

## Chunk 6: Finalize

### Task 8: String replacement verification and commit

**Files:**
- Delete: `Loan_Prediction_Notebook_ipynb.ipynb`
- Delete: `build_notebook.py` (temp script)

- [ ] **Step 1: Verify no Colab-specific strings remain in the new notebook**

Run:
```bash
python -c "
import json
nb = json.load(open('Loan_Prediction_Notebook.ipynb', encoding='utf-8'))
src_all = ' '.join(''.join(c['source']) for c in nb['cells'])
checks = {
    'no google.colab': 'google.colab' not in src_all,
    'no drive.mount': 'drive.mount' not in src_all,
    'no !pip': '!pip' not in src_all,
    'no lightgbm': 'lightgbm' not in src_all,
    'no /content/drive': '/content/drive' not in src_all,
    'no duplicate read_csv train': src_all.count(\"read_csv\") == 2,
    'has DATA_DIR': 'DATA_DIR' in src_all,
    'has CV_FOLDS': 'CV_FOLDS' in src_all,
    'has RANDOM_STATE': 'RANDOM_STATE' in src_all,
    'has import os': 'import os' in src_all,
    'has RandomForestClassifier import': 'from sklearn.ensemble import RandomForestClassifier' in src_all,
    'run_interpretability takes args': 'run_interpretability_analysis(model, X, y)' in src_all,
}
for check, passed in checks.items():
    print(f'  {\"PASS\" if passed else \"FAIL\"}: {check}')
"
```

Expected: All checks show `PASS`.

If any check fails: fix the relevant cell content in the notebook before proceeding.

- [ ] **Step 2: Verify read_csv count**

The new notebook should have exactly 2 `pd.read_csv` calls (one for train, one for test in Section 2). The check `'no duplicate read_csv train': src_all.count("read_csv") == 2` in Step 1 confirms this. If it shows FAIL, locate and remove the extra load.

- [ ] **Step 3: Remove the old notebook**

```bash
git rm "Loan_Prediction_Notebook_ipynb.ipynb"
```

- [ ] **Step 4: Remove the temp builder script**

```bash
rm build_notebook.py
```

- [ ] **Step 5: Stage new notebook and commit everything**

```bash
git add Loan_Prediction_Notebook.ipynb
git commit -m "refactor: port notebook to portable 7-section structure

- Remove all Google Colab dependencies (drive.mount, !pip install, hardcoded paths)
- Rename to Loan_Prediction_Notebook.ipynb
- Add CONFIG cell (DATA_DIR, RANDOM_STATE, CV_FOLDS)
- Reorganize into 7 labeled sections with markdown headers
- Update evaluate_with_cv signature: cv_splits iterator -> cv_folds integer
- Fix run_interpretability_analysis to accept explicit (model, X, y) params
- Remove duplicate pd.read_csv calls (reuse train_df and test_df from Section 2)
- Remove Cell 24 CODE HERE stub
- Add confusion matrix cell in Section 5
- Add feature importance plot in Section 5
- Add English docstrings to all functions
- Add inline comments for log transforms and risk feature engineering"
```
