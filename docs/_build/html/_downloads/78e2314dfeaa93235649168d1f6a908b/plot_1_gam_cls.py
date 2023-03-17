# -*- coding: utf-8 -*-
"""
Generalized Additive Model
=====================================

GAM Regression
"""

#%%
from piml import Experiment
from piml.models import GAMClassifier

exp = Experiment()
exp.data_loader(data="CoCircles")
exp.data_prepare(target='target', task_type='Classification', test_ratio=0.2)

#%%
# Model Training
exp.model_train(model=GAMClassifier(spline_order=2, n_splines=20, lam=0.6), name="GAM")

#%%
# Global interpretation: effect plot for X0
exp.model_interpret(model='GAM', show='global_effect_plot', uni_feature="X0", original_scale=True, figsize=(6, 5))

#%%
# Global interpretation: effect plot for X1
exp.model_interpret(model='GAM', show='global_effect_plot', uni_feature="X1", original_scale=True, figsize=(6, 5))

#%%
# Global interpretation: feature importance
exp.model_interpret(model='GAM',show='global_fi', figsize=(6, 5))

#%%
# Local interpretation
exp.model_interpret(model='GAM',show='local_fi', sample_id=0, figsize=(6, 5))
