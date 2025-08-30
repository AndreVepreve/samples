
# Introduction to Machine Learning (Hands-On, Interview-Ready)

> A pragmatic, deeply technical refresher you can use as a crash course before interviews or exams. It expands the corresponding chapter in your PDF and mirrors your repo’s structure. Examples use scikit-learn so you can paste them into notebooks.

---

## Table of Contents

1. What Machine Learning Means in Practice  
2. Data Splits, Cross-Validation & Leakage  
3. Feature Engineering & Preprocessing (without leaking)  
4. Core Model Zoo (you should master first)  
5. Under/Overfitting, Bias–Variance, and Regularization  
6. Model Selection & Hyperparameter Tuning  
7. Evaluation Metrics (classification, regression, calibration)  
8. Imbalanced Data: Practical Tactics  
9. Model Interpretation (Permutation Importance, SHAP overview)  
10. End-to-End Tabular Pipeline (copy/paste)  
11. Production Notes (persistence, reproducibility, monitoring)  
12. Common Pitfalls & Interview Traps  
13. Appendix: Learning/Validation Curves + Unsupervised Starter

---

## 1) What Machine Learning Means in Practice

- Learn a function \(f: \mathcal{X} \to \mathcal{Y}\) from data to minimize expected loss on *unseen* examples.
- No single model dominates: selection is empirical and task-dependent. Frame the **decision**, **metric**, **latency/cost** constraints first.

**Problem framing checklist**  
- Target definition (unit-of-prediction, horizon)  
- Data availability & staleness, feature lineage (avoid future info)  
- Metric(s) aligned to cost (e.g., PR-AUC > ROC-AUC on rare positives)  
- Offline/online split plan; retrain cadence & monitoring

---

## 2) Data Splits, Cross-Validation & Leakage

**Split early.** Hold out a test set before any preprocessing. Keep validation inside cross-validation (CV).

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # preserves class ratios
)
```

**CV essentials**  
- Use `StratifiedKFold` for classification; `KFold` otherwise.  
- Keep all preprocessing inside the CV loop via `Pipeline`/`ColumnTransformer` to prevent **data leakage**.

**Leakage examples**  
- Fitting scalers/encoders on all data.  
- Using target to derive features (target-aware encoding) outside a CV-safe scheme.  
- Temporal leakage (train sees future). Fix with time-based splits (`TimeSeriesSplit`).

---

## 3) Feature Engineering & Preprocessing (without leaking)

Use **pipelines** to bind transforms with the estimator, and **ColumnTransformer** to apply column-wise transforms.

```python
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

num_cols = ["age","income","score"]
cat_cols = ["country","device"]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

clf = Pipeline([
    ("pre", pre),
    ("model", LogisticRegression(max_iter=200))
])
clf.fit(X_train, y_train)
```

Notes:  
- Scaling helps gradient & distance-based models.  
- One-hot for low/medium-cardinality categoricals; leave high-cardinality to target/impact encoding with proper CV.

---

## 4) Core Model Zoo (you should master first)

**Linear models**: Linear/Logistic regression with L2 or L1; fast baselines; interpretable coefficients; well-calibrated with proper regularization.  
**k-Nearest Neighbors**: non-parametric; relies on scaling; sensitive to irrelevant features.  
**SVM/SVC**: large-margin; linear or RBF/kernelized for nonlinearity.  
**Decision Trees**: interpretable; require depth/leaf constraints.  
**Random Forests / ExtraTrees**: strong default on tabular; robust to scaling & monotonic transforms.  
**Gradient Boosting (XGBoost/LightGBM/CatBoost)**: powerful for structured data; mind tuning & early stopping.  
**KMeans (unsupervised)**: clustering by minimizing inertia; pick \(k\); use k-means++ init; standardize features.

---

## 5) Under/Overfitting, Bias–Variance, and Regularization

- **High bias**: underfit — both train/val errors high. Remedy: add capacity/features; reduce regularization.  
- **High variance**: overfit — train \(\ll\) val error. Remedy: more data/augmentation; stronger regularization; ensembles.  

**Regularization**  
- L2 (ridge): shrinks coefficients, stabilizes.  
- L1 (lasso): induces sparsity, useful for selection.  
- Early stopping acts like regularization for iterative learners.

**Learning curves** diagnose whether you need more data or more capacity (see Appendix).

---

## 6) Model Selection & Hyperparameter Tuning

Prefer **random search** over huge grids; then refine. Keep search **inside CV** to avoid optimism.

```python
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import loguniform

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_dist = {"model__C": loguniform(1e-3, 1e2)}
rnd = RandomizedSearchCV(clf, param_distributions=param_dist,
                         n_iter=40, cv=cv, n_jobs=-1, scoring="f1")
best = rnd.fit(X_train, y_train).best_estimator_
```

Tips  
- Use log-spaced ranges for scale-sensitive params (e.g., `C`, `alpha`).  
- Start broad (random search), then fine-tune with a small grid around the best region.

---

## 7) Evaluation Metrics (classification, regression, calibration)

**Classification**  
- Accuracy (balanced data only), Precision/Recall/F1, ROC-AUC, PR-AUC, Log-loss.  
- Always inspect the **confusion matrix** at candidate thresholds.  
- For rare positives, optimize PR-AUC or cost-weighted F\(_\beta\).

**Regression**  
- MAE (robust), MSE/RMSE, \(R^2\). Consider Huber/Pseudo-Huber for heavy tails.

**Calibration**  
- If acting on probabilities, calibrate: `CalibratedClassifierCV` with `method='isotonic'` or `'sigmoid'` (Platt).  
- Plot reliability with `calibration_curve`; track **Brier score**.

---

## 8) Imbalanced Data: Practical Tactics

- **Stratified splits**; don’t lose the minority class.  
- **Class weights** (`class_weight='balanced'` or custom).  
- **Resampling**: SMOTE/ADASYN (oversample) + undersample; evaluate with PR-AUC/AP; move threshold to desired precision/recall trade-off.  
- **Cost-sensitive** training when false negatives are expensive.

```python
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

smote_logreg = make_pipeline(
    SMOTE(k_neighbors=5, random_state=42),
    LogisticRegression(max_iter=500)
)
```

---

## 9) Model Interpretation (Permutation Importance, SHAP overview)

**Permutation importance**: model-agnostic; measures change in metric when a feature is permuted on held-out data.  
**SHAP**: Shapley-value-based local+global explanations; helpful for tree ensembles; requires extra dependency.

```python
from sklearn.inspection import permutation_importance
r = permutation_importance(best, X_test, y_test, n_repeats=20, scoring="roc_auc", random_state=42)
imp = (r.importances_mean, r.importances_std)
```

---

## 10) End-to-End Tabular Pipeline (copy/paste)

```python
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV

num_cols = X.select_dtypes(np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

base = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_dist = {
    "clf__n_estimators": [300, 400, 600],
    "clf__max_depth": [None, 8, 12, 20],
    "clf__max_features": ["sqrt", "log2", 0.5],
    "clf__min_samples_leaf": [1, 2, 5]
}

search = RandomizedSearchCV(base, param_distributions=param_dist,
                            n_iter=20, cv=cv, n_jobs=-1, scoring="roc_auc", random_state=42)
search.fit(X_train, y_train)

best = search.best_estimator_

cal = CalibratedClassifierCV(best, cv=3)   # isotonic by default
cal.fit(X_train, y_train)

pred = cal.predict(X_test)
proba = cal.predict_proba(X_test)[:, 1]

print(classification_report(y_test, pred))
print("ROC AUC:", roc_auc_score(y_test, proba))

joblib.dump(cal, "model.joblib")
```

---

## 11) Production Notes (persistence, reproducibility, monitoring)

- **Persistence**: save the *full pipeline* (preprocessing + model) with `joblib.dump`; load with matching library versions.  
- **Reproducibility**: set seeds, log versions, capture training code, parameters, and data snapshot.  
- **Monitoring**: track input drift (KS/PSI), output metrics (AUC/F1/MAE), and calibration; automate alerts & retraining.

---

## 12) Common Pitfalls & Interview Traps

1. Scaling/encoding **before** splitting (leakage).  
2. Using accuracy on imbalanced data; prefer PR-AUC/F1 and threshold tuning.  
3. Huge parameter grids; start with **random search** under a fixed budget.  
4. Not using `Pipeline`/`ColumnTransformer` — train/serve mismatches.  
5. Ignoring calibration when actions depend on probabilities.  
6. Forgetting to stratify classification splits.

---

## 13) Appendix

### A) Learning & Validation Curves

```python
from sklearn.model_selection import learning_curve, validation_curve
import numpy as np

# Learning curve (need more data or more capacity?)
sizes, train_scores, val_scores = learning_curve(
    best, X, y, cv=5, scoring="roc_auc", n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 6))

# Validation curve (sensitivity to a hyperparameter)
from sklearn.linear_model import LogisticRegression
param_range = np.logspace(-3, 2, 8)
tr, va = validation_curve(
    Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000))]),
    X, y, param_name="model__C", param_range=param_range,
    cv=5, scoring="f1", n_jobs=-1)
```

### B) Quick Unsupervised Starter (KMeans)

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

km = make_pipeline(StandardScaler(), KMeans(n_clusters=8, n_init="auto", random_state=42))
labels = km.fit_predict(X)
inertia = km[-1].inertia_
```

---

**Suggested filename:** `introduction_to_machine_learning.md`  
**Suggested repo location:** `samples/Week5/`
