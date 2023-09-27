# -*- coding: utf-8 -*-
"""
GAM Classification (CoCircles)
=====================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import GAMClassifier

exp = Experiment()
exp.data_loader(data="CoCircles", silent=True)
exp.data_prepare(target="target", task_type="classification", silent=True)

#%%
# Train Model
exp.model_train(model=GAMClassifier(spline_order=2, n_splines=20, lam=0.6), name="GAM")

#%%
# Evaluate predictive performance
exp.model_diagnose(model="GAM", show="accuracy_table")

#%%
# Global interpretation: effect plot for X0
exp.model_interpret(model="GAM", show="global_effect_plot", uni_feature="X0",
                    original_scale=True, figsize=(5, 4))

#%%
# Global interpretation: effect plot for X1
exp.model_interpret(model="GAM", show="global_effect_plot", uni_feature="X1",
                    original_scale=True, figsize=(5, 4))

#%%
# Global interpretation: feature importance
exp.model_interpret(model="GAM", show="global_fi", figsize=(5, 4))

#%%
# Local interpretation
exp.model_interpret(model="GAM", show="local_fi", sample_id=0, original_scale=True, figsize=(5, 4))
