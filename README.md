# XGBoost Regression — Encoding & Predictions

This repository contains a compact, notebook-driven workflow to (1) encode tabular data, (2) train an XGBoost regression model, and (3) generate predictions for a hold‑out/test set.

> If you just want to reproduce the final predictions, see **Usage** below.

---

## Contents

```
.
├── ENCODING.ipynb        # Data preparation & feature encoding
├── XGBoost.ipynb         # Model training & inference (XGBoost regressor)
├── Encoded_Train.csv     # Model-ready training data (features + target)
├── Encoded_Test.csv      # Model-ready test data (features only)
├── predictions.txt       # Model predictions exported from XGBoost.ipynb
└── README.md             # This file
```

---

## Quick Start

### 1) Set up the environment

You can use any recent Python (3.9–3.12). The project relies on common PyData packages plus XGBoost.

```bash
# Option A: install packages system-wide or in a venv
pip install -U jupyter pandas numpy scikit-learn xgboost matplotlib

# Option B (recommended): create an isolated environment via conda
conda create -y -n xgb-env python=3.11
conda activate xgb-env
pip install -U jupyter pandas numpy scikit-learn xgboost matplotlib
```

### 2) Open the notebooks

```bash
jupyter lab
# or
jupyter notebook
```

Then run the notebooks in the following order:

1. **ENCODING.ipynb**  
   - Loads raw/structured inputs (here we work directly with the already-encoded CSVs).  
   - Ensures features are numeric and aligned across train/test.  
   - Writes the artifacts: `Encoded_Train.csv` and `Encoded_Test.csv`.

2. **XGBoost.ipynb**  
   - Loads `Encoded_Train.csv` and `Encoded_Test.csv`.  
   - Trains an **XGBoost Regressor** (edit hyperparameters in the notebook).  
   - Optionally performs a validation split and prints metrics.  
   - Generates predictions on the test data and writes `predictions.txt` (one value per row).

> Tip: You can re-run both notebooks end‑to‑end via CLI automation (e.g., with [papermill](https://papermill.readthedocs.io/)) if you want reproducible builds in CI.

---

## Usage

### Train and evaluate
Open **XGBoost.ipynb**, adjust the model parameters in the training cell (e.g., `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`), then **Run All**. If a validation split is included, the notebook prints standard regression metrics (e.g., MAE/RMSE/R²).

### Generate predictions
After training, the notebook predicts for `Encoded_Test.csv` and exports a newline‑delimited file at `predictions.txt`. You can commit this file as the competition/assignment submission or a downstream input for other systems.

---

## Data expectations

- `Encoded_Train.csv` should contain **all numeric features** plus a **target** column. If your target is named differently, update the relevant cell in the notebook.  
- `Encoded_Test.csv` should contain the **same feature columns** as train (no target).  
- If you add or drop features, ensure both CSVs stay column‑aligned (order and names).

---

## Reproducibility

- Set the random seed(s) in the training cell for stable splits and model training.  
- Pin exact library versions in a `requirements.txt` or `environment.yml` if you need byte‑for‑byte reproducibility in CI. Example:

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
xgboost==2.0.3
matplotlib==3.8.4
```

---

## Troubleshooting

- **Shape or column mismatch**: Confirm that train and test have identical feature columns (names, order, dtype).  
- **NaNs after encoding**: Impute or drop rows/columns before training; XGBoost can handle some missing data, but consistency matters.  
- **Overfitting**: Reduce `max_depth`, increase `reg_lambda`/`reg_alpha`, or use early stopping with a validation set.

---

## How to adapt

- Swap the model: try `RandomForestRegressor`, `LightGBM`, or `CatBoost`.  
- Add feature scaling: for some models (e.g., linear), use `StandardScaler`.  
- Add cross‑validation: wrap training in `KFold` or `GroupKFold` to stabilize metrics.

---

## License

Choose a license (e.g., MIT) and add it as `LICENSE` in the repository if you intend to share or publish this work.
