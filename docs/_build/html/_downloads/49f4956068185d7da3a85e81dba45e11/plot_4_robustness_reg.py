# -*- coding: utf-8 -*-
"""
Robustness:  Regression
=====================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import FIGSRegressor

exp = Experiment()
exp.data_loader(data="BikeSharing", silent=True)
exp.data_summary(feature_exclude=["yr", "mnth", "temp"], silent=True)
exp.data_prepare(target="cnt", task_type="regression", silent=True)

#%%
# Train Model
exp.model_train(FIGSRegressor(max_iter=100, max_depth=5), name="FIGS")

#%%
# Robustness test with all features being perturbed
exp.model_diagnose(model="FIGS", show='robustness_perf', perturb_features=None,
                   perturb_method="raw", metric="MSE", perturb_size=0.1, figsize=(6, 4))

#%%
# Robustness test with custom perturbation features
exp.model_diagnose(model="FIGS", show="robustness_perf", perturb_features=["hr", "atemp"],
                   perturb_method='raw', metric="MSE", perturb_size=0.1,  figsize=(6, 4))

#%%
# Robustness test with custom perturbation size
exp.model_diagnose(model="FIGS", show="robustness_perf", perturb_features=None,
                   perturb_method="raw", metric="MSE", perturb_size=0.2, figsize=(6, 4))

#%%
# Robustness test with custom perturbation method
exp.model_diagnose(model="FIGS", show="robustness_perf", perturb_features=None,
                   perturb_method="quantile", metric="MSE", perturb_size=0.1, figsize=(6, 4))

#%%
# Robustness test with custom metrics
exp.model_diagnose(model="FIGS", show="robustness_perf", perturb_features=None,
                   perturb_method="raw", metric="R2", perturb_size=0.1, figsize=(6, 4))

#%%
# Robustness test on worst-alpha percent samples
exp.model_diagnose(model="FIGS", show="robustness_perf_worst", perturb_features=None,
                   perturb_method="raw", metric="MSE", perturb_size=0.1, alpha=0.3, figsize=(6, 4))
