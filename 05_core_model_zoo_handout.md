
# Core Model Zoo — Interview Handout & Hyperparameter Cheat Sheets

> A compact, printable reference for the models in your Week 5 chapter. Each sheet includes: when to use, scale sensitivity, key knobs, **sensible starting ranges**, a minimal code snippet, and pitfalls. Use these as baseline grids for `RandomizedSearchCV` → refine with a small grid near the best region.

---

## Linear & Logistic Regression (with Regularization)

**Definition.** Linear regression models a continuous target as a linear combination of features; ridge adds an L2 penalty, lasso adds an L1 penalty, and elastic‑net mixes both. Logistic regression is a generalized linear model that maps \(w^\top x\) through the logistic function to model class probabilities.

**Purpose.** Fast, interpretable baselines: use linear/ridge/lasso for continuous targets (robust to multicollinearity with ridge; feature selection with lasso) and logistic for classification with calibrated probabilities and sparse/high‑dimensional features.


**When**: fast, strong baselines; interpretable; handles high‑dimensional sparse data.  
**Scale**: standardize features for stability (especially with penalties).

**Key knobs**
- **Linear (Ridge/Lasso/ElasticNet)**: `alpha` (L2 strength), `l1_ratio` (for ElasticNet).  
- **Logistic**: `C` (inverse regularization), `penalty`, `solver`, `class_weight`.

**Starting ranges**
```python
# Linear regression with regularization
from sklearn.linear_model import Ridge, Lasso, ElasticNet
ridge_grid = {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}
lasso_grid = {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]}
enet_grid  = {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10], "l1_ratio": [0.1, 0.5, 0.9]}

# Logistic regression
from scipy.stats import loguniform
logreg_dist = {"model__C": loguniform(1e-3, 1e2), "model__penalty": ["l2"], "model__solver": ["lbfgs", "saga"]}
```
**Minimal code**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
clf = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000))])
```
**Pitfalls**: incompatible (`penalty`, `solver`) combos; unscaled features; extreme `C` leading to overfit/underfit.

---

## k‑Nearest Neighbors (kNN)

**Definition.** A non‑parametric method that predicts using the labels/values of the \(k\) closest training points under a chosen distance metric.

**Purpose.** Simple baseline when locality captures the concept of similarity; useful on small/medium data after scaling, or as a yardstick for more complex models.


**When**: local similarity matters; small/medium datasets.  
**Scale**: **must** scale numeric features; consider cosine distance for text embeddings.

**Key knobs**: `n_neighbors`, `weights` (`uniform`/`distance`), `metric` (`minkowski` with `p`=1/2; or `cosine`).

**Starting ranges**
```python
knn_grid = {
  "model__n_neighbors": [3,5,7,11,15,21],
  "model__weights": ["uniform", "distance"],
  "model__p": [1,2],  # L1 vs L2
}
```
**Minimal code**
```python
from sklearn.neighbors import KNeighborsClassifier
knn = Pipeline([("pre", pre), ("model", KNeighborsClassifier())])
```

**Pitfalls**: high prediction cost; curse of dimensionality (reduce features or use metric learning).

---

## Support Vector Machines (SVC)

**Definition.** Maximum‑margin classifiers that separate classes with the widest possible margin; kernels (e.g., RBF) lift inputs to non‑linear feature spaces for flexible decision boundaries.

**Purpose.** Strong choice for high‑dimensional problems and non‑linear boundaries on moderate‑sized datasets; offers robust performance with careful scaling and tuning of \(C\) and \(\gamma\).


**When**: high‑dimensional problems; non‑linear decision boundaries (RBF).  
**Scale**: **always** scale features.

**Key knobs**: `C` (margin vs. violations), `kernel` (e.g., `"rbf"`), `gamma` (RBF width), `class_weight`.

**Starting ranges**
```python
svc_dist  = {"model__C": loguniform(1e-2, 1e2),
             "model__gamma": loguniform(1e-4, 1e0),
             "model__kernel": ["rbf"]}
```
**Minimal code**
```python
from sklearn.svm import SVC
svc = Pipeline([("pre", pre), ("model", SVC(kernel="rbf", probability=True))])
```

**Pitfalls**: quadratic+ training complexity on large n; poor calibration without `probability=True` or post‑calibration.

---

## Decision Trees

**Definition.** Recursive partitioning that splits the feature space into axis‑aligned regions; predictions are piecewise‑constant within leaves.

**Purpose.** Interpretable rules and automatic interaction handling; foundation learners for powerful tree ensembles.


**When**: need simple rules/interpretability; non‑linear tabular signals.  
**Scale**: insensitive to monotonic transforms; no scaling required.

**Key knobs**: `max_depth`, `min_samples_leaf`, `max_features`, `ccp_alpha` (pruning).

**Starting ranges**
```python
tree_grid = {
  "model__max_depth": [None, 4, 6, 8, 12],
  "model__min_samples_leaf": [1, 2, 5, 10],
  "model__max_features": ["sqrt", "log2", None]
}
```
**Minimal code**
```python
from sklearn.tree import DecisionTreeClassifier
tree = Pipeline([("pre", pre), ("model", DecisionTreeClassifier(random_state=42))])
```

**Pitfalls**: overfitting without depth/leaf constraints; unstable splits on tiny datasets.

---

## Random Forests

**Definition.** An ensemble of decision trees trained on bootstrapped samples with random feature sub‑sampling; predictions are averaged to reduce variance.

**Purpose.** Strong, out‑of‑the‑box baseline for tabular data with mixed feature types; robust to scaling and outliers with minimal tuning.


**When**: robust, strong default on tabular data; good first non‑linear model.  
**Scale**: no scaling required; handles mixed feature types.

**Key knobs**: `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`, `bootstrap`.

**Starting ranges**
```python
rf_grid = {
  "clf__n_estimators": [300, 500, 800],
  "clf__max_depth": [None, 8, 12, 20],
  "clf__max_features": ["sqrt", "log2", 0.5],
  "clf__min_samples_leaf": [1, 2, 5],
  "clf__bootstrap": [True]
}
```
**Minimal code**
```python
from sklearn.ensemble import RandomForestClassifier
rf = Pipeline([("pre", pre),
               ("clf", RandomForestClassifier(random_state=42, n_jobs=-1))])
```

**Pitfalls**: too‑deep trees → overfit; impurity importances biased toward high‑cardinality categoricals (prefer permutation importance).

---

## Extra Trees (Extremely Randomized Trees)

**Definition.** An ensemble like random forests but with extra randomization: split thresholds are drawn at random per feature and the best among them is chosen, further de‑correlating trees.

**Purpose.** Fast, variance‑reducing alternative to random forests that often performs competitively on noisy tabular problems.


**When**: noisy data; want lower variance vs single trees; very fast.  
**Scale**: not required.

**Key knobs**: similar to RF; more randomness in splits.  
**Starting ranges**
```python
et_grid = {
  "clf__n_estimators": [400, 800],
  "clf__max_depth": [None, 8, 12],
  "clf__max_features": ["sqrt", "log2", 0.5],
  "clf__min_samples_leaf": [1, 2, 5]
}
```
**Minimal code**
```python
from sklearn.ensemble import ExtraTreesClassifier
et = Pipeline([("pre", pre),
               ("clf", ExtraTreesClassifier(random_state=42, n_jobs=-1))])
```

**Pitfalls**: use as an **ensemble** (single ExtraTree is too random).

---

## Gradient Boosting (Histogram‑based GBDT)

**Definition.** Gradient‑boosted decision trees fitted stage‑wise to the negative gradient of a chosen loss; the histogram‑based variant bins features to speed up training on large datasets.

**Purpose.** High‑accuracy choice for many structured/tabular tasks with strong non‑linearities; supports early stopping and regularization for precise control.


**When**: best‑in‑class for many structured/tabular problems; supports monotonic constraints in recent versions.  
**Scale**: not required; handles missing values internally (HistGB).

**Key knobs**: `learning_rate`, `max_leaf_nodes` (or `max_depth`), `n_estimators`, `min_samples_leaf`, `l2_regularization`, `subsample`. Enable `early_stopping=True`.

**Starting ranges**
```python
hgb_dist = {
  "model__learning_rate": [0.02, 0.05, 0.1],
  "model__max_leaf_nodes": [15, 31, 63],
  "model__l2_regularization": [0.0, 0.1, 1.0],
  "model__min_samples_leaf": [20, 50],
  "model__max_depth": [None]
}
```
**Minimal code**
```python
from sklearn.ensemble import HistGradientBoostingClassifier
gb = Pipeline([("pre", pre),
               ("model", HistGradientBoostingClassifier(early_stopping=True, random_state=42))])
```

**Pitfalls**: too‑large learning rates; no early stopping; insufficient trees (underfit).

---

## K‑Means (Unsupervised)

**Definition.** Centroid‑based clustering that partitions data into \(k\) clusters by minimizing within‑cluster sum of squares (inertia) using Lloyd’s iterative refinement; modern implementations use k‑means++ seeding.

**Purpose.** Quick segmentation and prototype discovery; often used for exploratory analysis, pre‑labeling, or building user/item cohorts.
**When**: prototype clustering; roughly spherical clusters; scalable.  
**Scale**: **must** scale features.

**Key knobs**: `n_clusters`, `init='k-means++'` (default), `n_init='auto'`, `max_iter`.

**Starting ranges**
```python
from scipy.stats import randint
kmeans_grid = {"model__n_clusters": [4, 6, 8, 10],
               "model__n_init": ["auto"], "model__max_iter": [200, 400]}
```
**Minimal code**
```python
from sklearn.cluster import KMeans
km = Pipeline([("scale", StandardScaler()), ("model", KMeans(n_clusters=8, n_init="auto", random_state=42))])
labels = km.fit_predict(X)
```

**Pitfalls**: local minima (use multiple inits), sensitivity to scaling/outliers, poor shape fit for non‑convex clusters (use DBSCAN/mean‑shift/spectral as alternatives).

---

## Tuning Playbook (copy/paste)

```python
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = RandomizedSearchCV(pipe, param_distributions=dist_or_grid,
                            n_iter=40, cv=cv, n_jobs=-1,
                            scoring="f1" or "roc_auc", random_state=42)
search.fit(X_train, y_train)
best = search.best_estimator_
```

**Rules of thumb**
- Start with **random search**, then a tiny grid around the winner.
- Use **log‑spaced** ranges for scale‑sensitive params (`C`, `alpha`, `gamma`). 
- Keep preprocessing inside the pipeline to avoid **leakage**.
- For probabilities you act on, add **isotonic** or **Platt** calibration and plot a reliability diagram.

---

**Suggested filename:** `core_model_zoo_handout.md`  
**Suggested repo location:** `samples/Week5/` (keep the document title independent of calendar terms).