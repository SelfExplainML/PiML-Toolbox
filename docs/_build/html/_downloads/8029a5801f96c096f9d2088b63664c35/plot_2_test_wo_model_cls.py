# -*- coding: utf-8 -*-
"""
Scored Test (Classification)
========================================================
Testing External Models without Model Objects as Input
"""

#%%
# Assume we have models fitted outside PiML workflow
import numpy as np
from xgboost import XGBClassifier
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

x, y = make_circles(n_samples=2000, noise=0.1, random_state=0)
train_x, test_x, train_y, test_y, train_idx, test_idx = train_test_split(x, y,
                                                                         np.arange(x.shape[0]), test_size=0.2)

xgb2 = XGBClassifier(max_depth=2, n_estimators=100)
xgb2.fit(train_x, train_y)

#%%
# Prepare the data for testing 
random_state = 0
target_name = "target"
task_type = 'classification'
feature_names = ["X1", "X2"]
feature_types = ["numerical", "numerical"] # "categorical" or "numerical"
prediction = xgb2.predict(x)
prediction_proba = xgb2.predict_proba(x)[:, 1]

data_params = {'x': x,
               'y': y,
               'prediction': prediction,
               'prediction_proba': prediction_proba,
               'feature_names': feature_names,
               'feature_types': feature_types,
               'target_name': target_name,
               'train_idx': train_idx,
               'test_idx': test_idx,
               'task_type': task_type,
               'random_state': random_state}

#%%
# Show the accuracy table 
from piml.scored_test import test_accuracy_table
res = test_accuracy_table(**data_params)

#%%
# Plot confusion matrix, ROC and Recall-Precision, only supports classifiers
from piml.scored_test import test_accuracy_plot
res = test_accuracy_plot(**data_params, figsize=(10, 4))

#%%
# Plot the prediction residuals against one feature of interest 
from piml.scored_test import test_accuracy_residual
res = test_accuracy_residual(**data_params, show_feature='X1', figsize=(6, 5))

#%%
# Run weakspot test to detect weak regions
from piml.scored_test import test_weakspot
res = test_weakspot(**data_params, slice_features=['X1'], figsize=(6, 5))

#%%
# Run overfit test to detect overfit regions
from piml.scored_test import test_overfit
res = test_overfit(**data_params, slice_features=['X1'], figsize=(6, 5))

#%%
# Run reliability diagram 
from piml.scored_test import test_reliability_perf
res = test_reliability_perf(**data_params, alpha=0.1, bins=10, figsize=(6, 5))

#%%
# Run reliability test to show the distributional distance between reliable and unreliable samples. 
from piml.scored_test import test_reliability_calibration
res = test_reliability_calibration(**data_params, alpha=0.1, figsize=(6, 5))

#%%
# Run reliability test to show the average coverage of prediction intervals 
from piml.scored_test import test_reliability_table
res = test_reliability_table(**data_params, alpha=0.1)

#%%
# Run reliability test to show the distributional distance between reliable and unreliable samples. 
from piml.scored_test import test_reliability_distance
res = test_reliability_distance(**data_params, alpha=0.1, threshold=1.1, figsize=(6, 5))

#%%
# Run reliability test to show relationship between prediction interval width and the feature of interest 
from piml.scored_test import test_reliability_marginal
res = test_reliability_marginal(**data_params, alpha=0.1, threshold=1.1, show_feature='X1',
                                bins=10, figsize=(6, 5))

#%%
# Run resilience test to show how model performance changes under distributional shift.
from piml.scored_test import test_resilience_perf
res = test_resilience_perf(**data_params, figsize=(6, 5))

#%%
# Run resilience test to show the distributional distance between worst samples and remaining samples.
from piml.scored_test import test_resilience_distance
res = test_resilience_distance(**data_params, figsize=(6, 5))

#%%
# Run resilience test to show how model performance changes under distributional shift (density plot).
from piml.scored_test import test_resilience_shift_density
res = test_resilience_shift_density(**data_params, alpha=0.3, metric='AUC', show_feature='X1',
                                    figsize=(6, 5))

#%%
# Run resilience test to show how model performance changes under distributional shift (histogram plot).
from piml.scored_test import test_resilience_shift_histogram
res = test_resilience_shift_histogram(**data_params, alpha=0.3, metric='AUC', show_feature='X1',
                                      figsize=(6, 5))
