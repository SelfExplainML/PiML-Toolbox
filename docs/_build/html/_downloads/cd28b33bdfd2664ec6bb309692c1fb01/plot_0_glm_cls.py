
# -*- coding: utf-8 -*-
"""
GLM Logistic Regression (Taiwan Credit)
==========================================

"""
#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import GLMClassifier

exp = Experiment()
exp.data_loader(data="TaiwanCredit", silent=True)
exp.data_summary(feature_exclude=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"], silent=True)
exp.data_prepare(target="FlagDefault", task_type="classification", silent=True)

#%%
# Train Model
exp.model_train(model=GLMClassifier(), name="GLM")

#%%
# Evaluate predictive performance
exp.model_diagnose(model="GLM", show='accuracy_table')

#%%
# Regression coefficient plot for numerical features
exp.model_interpret(model="GLM", show="glm_coef_plot", figsize=(5, 4))

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
exp.model_interpret(model="GLM", show="local_fi", sample_id=0, centered=False, original_scale=True, figsize=(5, 4))

#%%
# Local interpretation with centering and original scale of x
exp.model_interpret(model="GLM", show="local_fi", sample_id=0, centered=True, original_scale=True, figsize=(5, 4))
