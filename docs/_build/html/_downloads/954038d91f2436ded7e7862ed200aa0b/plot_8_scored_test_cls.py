"""
Scored Test: Classification
=====================================
"""

#%% Fit a PiML model to prepare the require data for scored test
import numpy as np
from piml import Experiment
from piml.models import XGB2Classifier

exp = Experiment()
exp.data_loader(data='TaiwanCredit', silent=True)
exp.data_prepare(silent=True)
exp.model_train(model=XGB2Classifier(), name='XGB2')

#%%
# Extract the required data inputs from PiML workflow 
train_x, train_y, _ = exp.get_data(train=True)
test_x, test_y, _ = exp.get_data(test=True)

task_type = 'classification'
X = np.concatenate([train_x, test_x], axis=0)
y = np.concatenate([train_y, test_y], axis=0).ravel()
prediction = exp.get_model("XGB2").estimator.predict(X)
prediction_proba = exp.get_model("XGB2").estimator.predict_proba(X)[:, -1]
feature_names = exp.get_feature_names()
feature_types = exp.get_feature_types()
target_name = exp.get_target_name()
train_idx = np.arange(train_x.shape[0])
test_idx = np.arange(train_x.shape[0], train_x.shape[0] + test_x.shape[0])

#%%
# Prepare the necessary data information
data_dict = {'x': X,
             'y': y,
             'task_type': task_type,
             'prediction': prediction,
             'prediction_proba': prediction_proba,
             'feature_names': feature_names,
             'feature_types': feature_types,
             'target_name': target_name,
             'train_idx': train_idx,
             'test_idx': test_idx}

#%%
# Show the accuracy table 
from piml.scored_test import test_accuracy_table
result = test_accuracy_table(**data_dict)

#%%
# Show the accuracy plot 
from piml.scored_test import test_accuracy_plot
result = test_accuracy_plot(**data_dict, figsize=(10, 4))

#%%
# Plot the prediction residuals against one feature of interest 
from piml.scored_test import test_accuracy_residual
result = test_accuracy_residual(**data_dict, show_feature='PAY_1', figsize=(5, 4))

#%%
# Run weakspot test to detect weak regions
from piml.scored_test import test_weakspot
result = test_weakspot(**data_dict, slice_features=['PAY_1'], figsize=(5, 4))

#%%
# Run overfit test to detect overfit regions
from piml.scored_test import test_overfit
result = test_overfit(**data_dict, slice_method="histogram", slice_features=['PAY_1'], figsize=(5, 4))

#%%
# Run reliability test to get reliability diagram
from piml.scored_test import test_reliability_perf
result = test_reliability_perf(**data_dict, figsize=(5, 4))

#%%
# Run reliability test to show relationship between prediction uncertainty and feature of interest 
from piml.scored_test import test_reliability_marginal
result = test_reliability_marginal(**data_dict, show_feature='PAY_1', figsize=(5, 4))

#%%
# Run reliability test to show data distance between reliable and unreliable samples. 
from piml.scored_test import test_reliability_distance
result = test_reliability_distance(**data_dict, figsize=(5, 4))

#%%
# Run reliability test to get the calibrated predicted probability vs. original predicted probability. 
from piml.scored_test import test_reliability_calibration
result = test_reliability_calibration(**data_dict, figsize=(5, 4))

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
result = test_resilience_shift_histogram(**data_dict, show_feature='PAY_1', figsize=(5, 4))

#%%
# The distributional difference density plot. 
from piml.scored_test import test_resilience_shift_density
result = test_resilience_shift_density(**data_dict, show_feature='PAY_1', figsize=(5, 4))
