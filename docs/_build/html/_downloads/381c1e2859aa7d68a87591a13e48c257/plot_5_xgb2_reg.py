# -*- coding: utf-8 -*-
"""
XGB-2 Regression (Bike Sharing)
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
# Evaluate predictive performance
exp.model_diagnose(model="XGB2", show="accuracy_table")

#%%
# Global effect plot for season
exp.model_interpret(model="XGB2", show="global_effect_plot", uni_feature="season", figsize=(5, 4))

#%%
# Global effect plot for hr
exp.model_interpret(model="XGB2", show="global_effect_plot", uni_feature="atemp", original_scale=True, figsize=(5, 4))

#%%
# Global effect plot for hr and season
exp.model_interpret(model="XGB2", show="global_effect_plot", bi_features=["hr", "season"],
                    sliced_line=False, original_scale=True, figsize=(5, 4))

#%%
# Effect importance
exp.model_interpret(model="XGB2", show="global_ei", figsize=(5, 4))

#%%
# Feature importance
exp.model_interpret(model="XGB2", show="global_fi", figsize=(5, 4))

#%%
# Local interpretation by effect
exp.model_interpret(model="XGB2", show="local_ei", sample_id=0, original_scale=True, figsize=(5, 4))

#%%
# Local interpretation by feature
exp.model_interpret(model="XGB2", show="local_fi", sample_id=0, original_scale=True, figsize=(5, 4))
