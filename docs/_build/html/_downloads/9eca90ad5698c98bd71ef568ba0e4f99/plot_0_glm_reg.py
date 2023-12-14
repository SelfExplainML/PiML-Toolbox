# -*- coding: utf-8 -*-
"""
GLM Linear Regression (Bike Sharing)
=====================================
"""
#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import GLMRegressor

exp = Experiment()
exp.data_loader(data="BikeSharing", silent=True)
exp.data_summary(feature_exclude=["yr", "mnth", "temp"], silent=True)
exp.data_prepare(target="cnt", task_type="regression", silent=True)

#%%
# Train Model
exp.model_train(model=GLMRegressor(), name="GLM")

#%%
# Evaluate predictive performance
exp.model_diagnose(model="GLM", show='accuracy_table')

#%%
# Regression coefficient plot for numerical features
exp.model_interpret(model="GLM", show="glm_coef_plot", figsize=(5, 4))

#%%
# Regression coefficient plot for categorical features
exp.model_interpret(model="GLM", show="glm_coef_plot", uni_feature="season", figsize=(5, 4))

#%%
# Regression coefficient table for all features
exp.model_interpret(model="GLM", show="glm_coef_table")

#%%
# Feature importance plot
exp.model_interpret(model="GLM", show="global_fi", figsize=(5, 4))

#%%
# Local interpretation without centering
exp.model_interpret(model="GLM", show="local_fi", sample_id=0, centered=False, original_scale=False, figsize=(5, 4))

#%%
# Local interpretation with original scale of x
exp.model_interpret(model="GLM", show='local_fi', sample_id=0, centered=False, original_scale=True, figsize=(5, 4))

#%%
# Local interpretation with centering and original scale of x
exp.model_interpret(model="GLM", show='local_fi', sample_id=0, centered=True, original_scale=True, figsize=(5, 4))


#%%
# Global feature importance based on test data.
exp.model_interpret(model="GLM", show="global_fi", use_test=True, figsize=(5, 4))

#%%
# Local interpretation for test set data with index 0.
exp.model_interpret(model="GLM", show="local_fi", sample_id=0, use_test=True,
                    original_scale=True, figsize=(5, 4))
