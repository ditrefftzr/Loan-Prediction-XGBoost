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
