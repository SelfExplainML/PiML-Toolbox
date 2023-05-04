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
# 1D PDP for hr
exp.model_explain(model="XGB2", show="pdp", uni_feature="hr",
                  original_scale=True, figsize=(5, 4))
#%%
# 1D PDP for season
exp.model_explain(model="XGB2", show="pdp", uni_feature='season',
                  original_scale=True, figsize=(5, 4))
#%%
# 2D PDP for hr and workingday
exp.model_explain(model="XGB2", show="pdp", bi_features=["hr", "workingday"],
                  pdp_size=10000, original_scale=True, figsize=(5, 4))
