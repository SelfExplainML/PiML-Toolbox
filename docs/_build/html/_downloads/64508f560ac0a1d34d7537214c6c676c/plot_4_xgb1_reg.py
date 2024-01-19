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
# Train model
exp.model_train(model=XGB1Regressor(n_estimators=500, max_bin=20, min_bin_size=0.01), name="XGB1")

#%%
# Train model with monotonicity constrained on MedInc
exp.model_train(model=XGB1Regressor(n_estimators=500, max_bin=20, min_bin_size=0.01,
                                    mono_increasing_list=("MedInc", )),
                name="Mono-XGB1")

#%%
# Evaluate predictive performance of XGB1
exp.model_diagnose(model="XGB1", show='accuracy_table')

#%%
# Evaluate predictive performance of Mono-XGB1
exp.model_diagnose(model="Mono-XGB1", show='accuracy_table')

#%%
# Global effect plot of XGB1 on MedInc
exp.model_interpret(model="XGB1", show="global_effect_plot", uni_feature="MedInc",
                    original_scale=True, figsize=(5, 4))

#%%
# Global effect plot of Mono-XGB1 on MedInc   
exp.model_interpret(model="Mono-XGB1", show="global_effect_plot", uni_feature="MedInc",
                    original_scale=True, figsize=(5, 4))

#%%
# Feature importance of Mono-XGB1    
exp.model_interpret(model="Mono-XGB1", show="global_fi", figsize=(5, 4))

#%%
# Weight of evidence plot of Mono-XGB1 on MedInc
exp.model_interpret(model="Mono-XGB1", show="xgb1_woe", uni_feature="MedInc", original_scale=True, figsize=(5, 4))

#%%
# Information value plot of Mono-XGB1  
exp.model_interpret(model="Mono-XGB1", show="xgb1_iv", figsize=(5, 4))

#%%
# Local interpretation of Mono-XGB1  
exp.model_interpret(model="Mono-XGB1", show="local_fi", sample_id=0, original_scale=True, figsize=(5, 4))
