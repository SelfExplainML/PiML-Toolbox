"""
Scored Test: Regression
=====================================
"""
#%% Fit a sklearn model to prepare the require data for scored test
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
# Extract the required data inputs from PiML workflow 
X = data.data
y = data.target
task_type = 'regression'
prediction = xgb2.predict(data.data)
feature_names = data.feature_names
target_name = data.target_names[0]

#%%
# Prepare the necessary data information
data_dict = {'x': X,
             'y': y,
             'task_type': task_type,
             'prediction': prediction,
             'feature_names': feature_names,
             'target_name': target_name,
             'train_idx': train_idx,
             'test_idx': test_idx}

#%%
# Show the accuracy table 
from piml.scored_test import test_accuracy_table
result = test_accuracy_table(**data_dict)

#%%
# Plot the prediction residuals against one feature of interest 
from piml.scored_test import test_accuracy_residual
result = test_accuracy_residual(**data_dict, show_feature='MedInc', figsize=(5, 4))

#%%
# Run weakspot test to detect weak regions
from piml.scored_test import test_weakspot
result = test_weakspot(**data_dict, slice_features=['MedInc'], figsize=(5, 4))

#%%
# Run overfit test to detect overfit regions
from piml.scored_test import test_overfit
result = test_overfit(**data_dict, slice_method="histogram", slice_features=['MedInc'], figsize=(5, 4))

#%%
# Run reliability test to show relationship between prediction uncertainty and feature of interest 
from piml.scored_test import test_reliability_marginal
result = test_reliability_marginal(**data_dict, show_feature='MedInc', figsize=(5, 4))

#%%
# Run reliability test to show data distance between reliable and unreliable samples. 
from piml.scored_test import test_reliability_distance
result = test_reliability_distance(**data_dict, figsize=(5, 4))

#%%
# Run resilience test to show how model performance changes under distributional shift.
from piml.scored_test import test_resilience_perf
result = test_resilience_perf(**data_dict, figsize=(5, 4))

#%%
# We can calculate the distributional difference between good regions and bad regions, e.g. the weak regions and the rest. Similarly, such plot can also be used for other tests, like reliablity. 
from piml.scored_test import test_resilience_distance
result = test_resilience_distance(**data_dict, figsize=(5, 4))

#%%
# The distributional difference histogram plot. 
from piml.scored_test import test_resilience_shift_histogram
result = test_resilience_shift_histogram(**data_dict, show_feature='MedInc', figsize=(5, 4))

#%%
# The distributional difference density plot. 
from piml.scored_test import test_resilience_shift_density
result = test_resilience_shift_density(**data_dict, show_feature='MedInc', figsize=(5, 4))
