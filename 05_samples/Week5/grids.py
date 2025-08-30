
from scipy.stats import loguniform

# Linear/Logistic
ridge_grid = {"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]}
lasso_grid = {"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]}
enet_grid  = {"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10], "model__l1_ratio": [0.1, 0.5, 0.9]}
logreg_dist = {"model__C": loguniform(1e-3, 1e2), "model__penalty": ["l2"], "model__solver": ["lbfgs", "saga"]}

# kNN
knn_grid = {"model__n_neighbors": [3,5,7,11,15,21], "model__weights": ["uniform","distance"], "model__p": [1,2]}

# SVC
svc_dist  = {"model__C": loguniform(1e-2, 1e2), "model__gamma": loguniform(1e-4, 1e0), "model__kernel": ["rbf"]}

# Decision Tree
tree_grid = {"model__max_depth": [None,4,6,8,12], "model__min_samples_leaf": [1,2,5,10], "model__max_features": ["sqrt","log2",None]}

# Random Forest
rf_grid = {"clf__n_estimators": [300,500,800], "clf__max_depth":[None,8,12,20], "clf__max_features":["sqrt","log2",0.5], "clf__min_samples_leaf":[1,2,5]}

# Extra Trees
et_grid = {"clf__n_estimators": [400,800], "clf__max_depth":[None,8,12], "clf__max_features":["sqrt","log2",0.5], "clf__min_samples_leaf":[1,2,5]}

# HistGradientBoosting
hgb_dist = {"model__learning_rate":[0.02,0.05,0.1], "model__max_leaf_nodes":[15,31,63], "model__l2_regularization":[0.0,0.1,1.0], "model__min_samples_leaf":[20,50], "model__max_depth":[None]}

# KMeans
kmeans_grid = {"model__n_clusters":[4,6,8,10], "model__n_init":["auto"], "model__max_iter":[200,400]}
