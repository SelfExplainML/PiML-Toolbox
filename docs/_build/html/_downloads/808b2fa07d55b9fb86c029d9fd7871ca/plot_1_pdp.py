# -*- coding: utf-8 -*-
"""
Partial Dependence Plot 
=====================================

"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import XGB2Regressor

exp = Experiment()
exp.data_loader(data="BikeSharing", silent=True)
exp.data_summary(feature_exclude=["yr", "mnth", "temp"], silent=True)
exp.data_prepare(target="cnt", task_type="regression", silent=True)

#%%
# Train Model
exp.model_train(model=XGB2Regressor(n_estimators=100), name="XGB2")

#%%
# 1D PDP for hr (use training data by default)
exp.model_explain(model="XGB2", show="pdp", uni_feature="hr",
                  grid_size=50, original_scale=True, figsize=(5, 4))

#%%
# 1D PDP for hr (use test data)
exp.model_explain(model="XGB2", show="pdp", uni_feature="hr", use_test=True,
                  grid_size=50, original_scale=True, figsize=(5, 4))

#%%
# 1D PDP for season
exp.model_explain(model="XGB2", show="pdp", uni_feature='season',
                  original_scale=True, figsize=(5, 4))

#%%
# 2D PDP for hr and workingday
exp.model_explain(model="XGB2", show="pdp", bi_features=["hr", "workingday"],
                  grid_size=10, sample_size=10000, sliced_line=False, original_scale=True, figsize=(5, 4))
