# -*- coding: utf-8 -*-
"""
Accuracy: Regression 
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
#Train Model
exp.model_train(model=XGB2Regressor(), name="XGB2")

#%%
#Accuracy table
exp.model_diagnose(model="XGB2", show="accuracy_table")
#%%
#Plot residual with respect to the feature hr
exp.model_diagnose(model="XGB2", show="accuracy_residual", show_feature="hr",
                   use_test=False, original_scale=True, figsize=(5, 4))
#%%
#Plot residual with respect to the feature season
exp.model_diagnose(model="XGB2", show="accuracy_residual", show_feature="season",
                   use_test=False, original_scale=True, figsize=(5, 4))
#%%
#Plot residual with respect to the target feature
exp.model_diagnose(model="XGB2", show="accuracy_residual", show_feature="cnt",
                   use_test=False, figsize=(5, 4))
#%%
#Plot residual with respect to the model prediction
exp.model_diagnose(model="XGB2", show="accuracy_residual", show_feature="cnt_predict",
                   use_test=False, figsize=(5, 4))
