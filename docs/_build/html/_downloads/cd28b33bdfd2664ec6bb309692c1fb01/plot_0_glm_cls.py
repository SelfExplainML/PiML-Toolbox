
# -*- coding: utf-8 -*-
"""
GLM - Logistic Regression Model
===================================

GLM Logistic Regression
"""
#%%
from piml import Experiment
from piml.models import GLMClassifier
exp = Experiment()
exp.data_loader(data='TaiwanCredit')
exp.data_summary(feature_exclude=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"], feature_type={})
exp.data_prepare(target='FlagDefault', task_type='Classification', test_ratio=0.2, random_state=0)

#%%
# Model training
exp.model_train(model=GLMClassifier(), name='GLM')
exp.model_diagnose(model='GLM', show='accuracy_table')

#%%
# Global interpretation
exp.model_interpret(model='GLM', show="glm_coef_plot")

exp.model_interpret(model='GLM', show="global_fi", figsize=(6, 5))
#%%
# Local interpretation
exp.model_interpret(model='GLM', show="local_fi", sample_id=0, centered=False, figsize=(6, 5))

exp.model_interpret(model="GLM", show='local_fi', sample_id=0, centered=False, original_scale=True, figsize=(6, 5))

exp.model_interpret(model="GLM", show='local_fi', sample_id=0, centered=True, original_scale=True, figsize=(6, 5))