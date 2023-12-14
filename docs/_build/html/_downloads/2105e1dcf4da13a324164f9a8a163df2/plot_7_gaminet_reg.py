# -*- coding: utf-8 -*-
"""
GAMI-Net Regression (Bike Sharing)
=====================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import GAMINetRegressor

exp = Experiment()
exp.data_loader(data="BikeSharing", silent=True)
exp.data_summary(feature_exclude=["yr", "mnth", "temp"], silent=True)
exp.data_prepare(target="cnt", task_type="regression", silent=True)

#%%
# Train Model
exp.model_train(model=GAMINetRegressor(), name="GAMI-Net")

#%%
# Evaluate predictive performance
exp.model_diagnose(model="GAMI-Net", show="accuracy_table")

#%%
# Global effect plot for hr and weekday
exp.model_interpret(model="GAMI-Net", show="global_effect_plot", bi_features=["hr", "weekday"],
                    original_scale=True, figsize=(5, 4))

#%%
# Global effect plot for hr
exp.model_interpret(model="GAMI-Net", show="global_effect_plot", uni_feature="hr",
                    original_scale=True, figsize=(5, 4))

#%%
# Global effect plot for weekday
exp.model_interpret(model="GAMI-Net", show="global_effect_plot", uni_feature="weekday", figsize=(5, 4))

#%%
# Effect importance
exp.model_interpret(model="GAMI-Net", show="global_ei", figsize=(5, 4))

#%%
# Feature importance
exp.model_interpret(model="GAMI-Net", show="global_fi", figsize=(5, 4))

#%%
# Local interpretation by effect
exp.model_interpret(model='GAMI-Net', show="local_ei", sample_id=0, original_scale=True, figsize=(5, 4))

#%%
# Local interpretation by feature
exp.model_interpret(model='GAMI-Net', show="local_fi", sample_id=0, original_scale=True, figsize=(5, 4))
