# -*- coding: utf-8 -*-
"""
Individual Conditional Expectation
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
exp.model_train(model=XGB2Regressor(), name="XGB2")

#%%
# ICE Plot for hr
exp.model_explain(model="XGB2", show="ice", uni_feature="hr", original_scale=True, figsize=(5, 4))

#%%
# ICE Plot for atemp
exp.model_explain(model="XGB2", show="ice", uni_feature="atemp", original_scale=True, figsize=(5, 4))

#%%
# ICE Plot for weekday
exp.model_explain(model="XGB2", show="ice", uni_feature="weekday", original_scale=True, figsize=(5, 4))
