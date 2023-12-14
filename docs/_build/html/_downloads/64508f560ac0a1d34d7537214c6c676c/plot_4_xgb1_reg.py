# -*- coding: utf-8 -*-
"""
XGB-1 Regression (California Housing)
=====================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import XGB1Regressor

exp = Experiment()
exp.data_loader(data="CaliforniaHousing_trim2", silent=True)
exp.data_prepare(target="MedHouseVal", task_type="regression", silent=True)

#%%
# Train Model
exp.model_train(model=XGB1Regressor(n_estimators=500, max_bin=20, min_bin_size=0.01), name="XGB1")

#%%
# Evaluate predictive performance
exp.model_diagnose(model="XGB1", show='accuracy_table')

#%%
# Global effect plot
exp.model_interpret(model="XGB1", show="global_effect_plot", uni_feature="MedInc",
                    original_scale=True, figsize=(5, 4))

#%%
# Feature importance
exp.model_interpret(model="XGB1", show="global_fi", figsize=(5, 4))

#%%
# Weight of evidence plot for MedInc
exp.model_interpret(model="XGB1", show="xgb1_woe", uni_feature="MedInc", original_scale=True, figsize=(5, 4))

#%%
# Information value plot
exp.model_interpret(model="XGB1", show="xgb1_iv", figsize=(5, 4))

#%%
# Local interpretation
exp.model_interpret(model="XGB1", show="local_fi", sample_id=0, original_scale=True, figsize=(5, 4))
