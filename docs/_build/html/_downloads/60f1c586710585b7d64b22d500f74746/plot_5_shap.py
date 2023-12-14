# -*- coding: utf-8 -*-
"""
SHapley Additive exPlanations
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
# SHAP Waterfall plot
exp.model_explain(model="XGB2", show="shap_waterfall", sample_id=0, figsize=(5, 4))
#%%
# SHAP feature importance
exp.model_explain(model="XGB2", show="shap_fi", sample_size=100, figsize=(5, 4))
#%%
# SHAP summary plot
exp.model_explain(model="XGB2", show="shap_summary", sample_size=100, figsize=(5, 4))
#%%
# SHAP scatter plot
exp.model_explain(model="XGB2", show="shap_scatter", uni_feature="hr", 
                  sample_size=100, figsize=(5, 4))
