# -*- coding: utf-8 -*-
"""
Scored Test (Regression)
========================================================
Testing External Models without Model Objects as Input
"""

#%%
# Assume we have models fitted outside PiML workflow
import numpy as np
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
train_x, test_x, train_y, test_y, train_idx, test_idx = train_test_split(data.data, data.target,
                                                                         np.arange(data.data.shape[0]), test_size=0.2)
xgb2 = XGBRegressor(max_depth=2, n_estimators=100)
xgb2.fit(train_x, train_y)

#%%
# Prepare the data for testing 
x = data.data
y = data.target
task_type = 'regression'
random_state = 0
feature_names = data.feature_names
feature_types = ["numerical", "numerical", "numerical", "numerical" "numerical", "numerical", "numerical", "numerical"] # "categorical" or "numerical"
target_name = data.target_names[0]
prediction = xgb2.predict(x)

data_params = {'x': x,
               'y': y,
               'prediction': prediction,
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
# Plot the prediction residuals against one feature of interest 
from piml.scored_test import test_accuracy_residual
res = test_accuracy_residual(**data_params, show_feature='MedInc', figsize=(6, 5))

#%%
# Run weakspot test to detect weak regions
from piml.scored_test import test_weakspot
res = test_weakspot(**data_params, slice_features=['MedInc'], figsize=(6, 5))

#%%
# Run overfit test to detect overfit regions
from piml.scored_test import test_overfit
res = test_overfit(**data_params, slice_features=['MedInc'], figsize=(6, 5))

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
res = test_reliability_marginal(**data_params, alpha=0.1, threshold=1.1, show_feature='MedInc',
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
res = test_resilience_shift_density(**data_params, alpha=0.3, metric='MAE', show_feature='MedInc',
                                    figsize=(6, 5))

#%%
# Run resilience test to show how model performance changes under distributional shift (histogram plot).
from piml.scored_test import test_resilience_shift_histogram
res = test_resilience_shift_histogram(**data_params, alpha=0.3, metric='MAE', show_feature='MedInc')
