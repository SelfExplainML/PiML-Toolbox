# -*- coding: utf-8 -*-
"""
Generalized Additive Model
=====================================

GAM Regression
"""

#%%
from piml import Experiment
from piml.models import GAMRegressor

exp = Experiment()
exp.data_loader(data="CaliforniaHousing_trim2")
exp.data_prepare()

#%%
# Model Training
exp.model_train(model=GAMRegressor(spline_order=1, n_splines=20, lam=0.6), name="GAM")

#%%
# Global interpretation: effect plot
exp.model_interpret(model='GAM', show='global_effect_plot', uni_feature="MedInc", original_scale=True, figsize=(6, 5))

#%%
# Global interpretation: feature importance
exp.model_interpret(model='GAM',show='global_fi', figsize=(6, 5))

#%%
# Local interpretation
exp.model_interpret(model='GAM',show='local_fi', sample_id=0, figsize=(6, 5))
