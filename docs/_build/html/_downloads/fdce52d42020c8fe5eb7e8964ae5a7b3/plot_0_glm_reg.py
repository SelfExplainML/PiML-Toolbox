
# -*- coding: utf-8 -*-
"""
GLM - Linear Regression Model
===================================

GLM Linear Regression
"""
#%%
from piml import Experiment
from piml.models import GLMRegressor
exp = Experiment()
exp.data_loader(data="BikeSharing")
exp.data_prepare(target='cnt', task_type='Regression', test_ratio=0.2, random_state=0)

#%%
# Model training
exp.model_train(model=GLMRegressor(), name='GLM')
exp.model_diagnose(model='GLM', show='accuracy_table')

#%%
# Global interpretation
exp.model_interpret(model='GLM', show="glm_coef_plot")

exp.model_interpret(model='GLM', show="glm_coef_plot", uni_feature="season")

exp.model_interpret(model='GLM', show="global_fi", figsize=(6, 5))
#%%
# Local interpretation
exp.model_interpret(model='GLM', show="local_fi", sample_id=0, centered=False, figsize=(6, 5))

exp.model_interpret(model="GLM", show='local_fi', sample_id=0, centered=False, original_scale=True, figsize=(6, 5))

exp.model_interpret(model="GLM", show='local_fi', sample_id=0, centered=True, original_scale=True, figsize=(6, 5))