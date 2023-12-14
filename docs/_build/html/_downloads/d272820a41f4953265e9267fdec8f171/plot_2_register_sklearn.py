# -*- coding: utf-8 -*-
"""
Registering sklearn Style Models
========================================================
Assume we have sklearn style models fitted outside PiML workflow
"""

#%%
# For demonstration, we fit a model using XGBoost's sklearn API
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data = fetch_california_housing()
train_x, test_x, train_y, test_y = train_test_split(data.data, data.target, test_size=0.2)
feature_names = data.feature_names
target_name = data.target_names[0]

xgb2 = XGBRegressor(max_depth=2, n_estimators=100)
xgb2.fit(train_x, train_y)

xgb7 = XGBRegressor(max_depth=7, n_estimators=100)
xgb7.fit(train_x, train_y)

#%%
# Load PiML
from piml import Experiment
exp = Experiment(highcode_only=True)

#%%
# Register the fitted model into PiML (please make sure the datasets of different pipelines are the same)
pipeline_xgb2 = exp.make_pipeline(model=xgb2,
                                  train_x=train_x,
                                  train_y=train_y.ravel(),
                                  test_x=test_x,
                                  test_y=test_y.ravel(),
                                  feature_names=feature_names,
                                  target_name=target_name)
exp.register(pipeline_xgb2, "XGB-External-2")

pipeline_xgb7 = exp.make_pipeline(model=xgb7,
                                  train_x=train_x,
                                  train_y=train_y.ravel(),
                                  test_x=test_x,
                                  test_y=test_y.ravel(),
                                  feature_names=feature_names,
                                  target_name=target_name)
exp.register(pipeline_xgb7, "XGB-External-7")

#%%
# Check model performance
exp.model_diagnose(model="XGB-External-2", show="accuracy_table")

#%%
# Check model performance
exp.model_diagnose(model="XGB-External-7", show="accuracy_table")

#%%
# Compare model robustness 
exp.model_compare(models=["XGB-External-2", "XGB-External-7"], show="robustness_perf", figsize=(5, 4))

#%%
# Compare model resilience
exp.model_compare(models=["XGB-External-7", "XGB-External-2"], show="resilience_perf", figsize=(5, 4))
