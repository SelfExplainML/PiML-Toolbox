# -*- coding: utf-8 -*-
"""
Accumulated Local Effects
=====================================

"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import ReluDNNRegressor

exp = Experiment()
exp.data_loader(data="BikeSharing", silent=True)
exp.data_summary(feature_exclude=["yr", "mnth", "temp"], silent=True)
exp.data_prepare(target="cnt", task_type="regression", silent=True)

#%%
# Train Model
exp.model_train(model=ReluDNNRegressor(), name="ReLUDNN")

#%%
# 1D ALE Plot for hr
exp.model_explain(model="ReLUDNN", show="ale", uni_feature='hr',
                  grid_size=50, original_scale=True, figsize=(5, 4))

#%%
# 1D ALE Plot for atemp
exp.model_explain(model="ReLUDNN", show="ale", uni_feature='atemp',
                  grid_size=50, original_scale=True, figsize=(5, 4))

#%%
# 1D ALE Plot for weathersit
exp.model_explain(model="ReLUDNN", show="ale", uni_feature='weathersit', 
                  original_scale=True, figsize=(5, 4))
#%%
# 2D ALE Plot for hr and atemp
exp.model_explain(model="ReLUDNN", show="ale", bi_features=["hr", "atemp"], 
                  grid_size=10, sliced_line=False, original_scale=True, figsize=(5, 4))
