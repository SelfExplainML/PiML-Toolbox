# -*- coding: utf-8 -*-
"""
GAM Regression (California Housing)
=====================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import GAMRegressor

exp = Experiment()
exp.data_loader(data="CaliforniaHousing_trim2", silent=True)
exp.data_prepare(target="MedHouseVal", task_type="regression", silent=True)

#%%
# Train Model
exp.model_train(model=GAMRegressor(spline_order=1, n_splines=20, lam=0.6), name="GAM")

#%%
# Evaluate predictive performance
exp.model_diagnose(model="GAM", show="accuracy_table")

#%%
# Global interpretation: effect plot
exp.model_interpret(model="GAM", show="global_effect_plot", uni_feature="MedInc",
                    original_scale=True, figsize=(5, 4))

#%%
# Global interpretation: feature importance
exp.model_interpret(model="GAM", show="global_fi", figsize=(5, 4))

#%%
# Local interpretation
exp.model_interpret(model="GAM", show="local_fi", sample_id=0,
                    original_scale=True, figsize=(5, 4))

#%%
# Local interpretation
exp.model_interpret(model="GAM", show="local_fi", sample_id=10,
                    original_scale=True, figsize=(5, 4))
