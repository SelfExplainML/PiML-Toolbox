# -*- coding: utf-8 -*-
"""
Testing External Models without Model Objects as Input
========================================================
"""

#%%
# Assume we have models fitted outside PiML workflow
import numpy as np
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

task_type = 'Regression'
data = fetch_california_housing()
train_x, test_x, train_y, test_y, train_idx, test_idx = train_test_split(data.data, data.target,
                                                                         np.arange(data.data.shape[0]), test_size=0.2)
feature_names = data.feature_names
target_name = data.target_names[0]

xgb2 = XGBRegressor(max_depth=2, n_estimators=100)
xgb2.fit(train_x, train_y)


#%%
# Prepare model prediction and necessary data information
data_dict = {'x': data.data,
             'y': data.target,
             'prediction': xgb2.predict(data.data),
             'feature_names': feature_names,
             'target_name': target_name,
             'train_idx': train_idx,
             'test_idx': test_idx}

#%%
# Show the accuracy table 
from piml.scored_test import test_accuracy
df = test_accuracy(**data_dict, task_type=task_type)

#%%
# Plot the prediction residuals against one feature of interest 
from piml.scored_test import residual_plot
result, fig = residual_plot(**data_dict, task_type=task_type, show_feature='MedInc')

#%%
# Run weakspot test to detect weak regions
from piml.scored_test import slicing_weakspot
result, fig = slicing_weakspot(**data_dict, task_type=task_type, slice_features=['MedInc'])

#%%
# Run overfit test to detect overfit regions
from piml.scored_test import slicing_overfit
result, fig = slicing_overfit(**data_dict, task_type=task_type, slice_method="histogram", slice_features=['MedInc'])

#%%
# Run reliability test to show relationship between prediction uncertainty and feature of interest 
from piml.scored_test import slicing_reliability
result, fig = slicing_reliability(**data_dict, task_type=task_type, slice_features=['MedInc'])

#%%
# Run resilience test to show how model performance changes under distributional shift.
from piml.scored_test import test_resilience
result, fig = test_resilience(**data_dict, task_type=task_type)

#%%
# We can calculate the distributional difference between good regions and bad regions, e.g. the weak regions and the rest. Similarly, such plot can also be used for other tests, like overfit, reliablity, etc. 
from piml.scored_test import two_sample_distance_weakspot
result, fig = two_sample_distance_weakspot(**data_dict, task_type=task_type, slice_features=['MedInc'], figsize=(6, 5))

#%%
# The distributional difference density plot. Similarly, such plot can also be used for other tests, like overfit, reliablity, etc. 
from piml.scored_test import marginal_shift_weakspot
result, fig = marginal_shift_weakspot(**data_dict, task_type=task_type, slice_features=['MedInc'],
                                       plot_type='density', show_feature='MedInc', figsize=(6, 5))
