# -*- coding: utf-8 -*-
"""
XGB-2 Classification (Taiwan Credit)
=====================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import XGB2Classifier

exp = Experiment()
exp.data_loader(data="TaiwanCredit", silent=True)
exp.data_summary(feature_exclude=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"], silent=True)
exp.data_prepare(target="FlagDefault", task_type="classification", silent=True)

#%%
# Train Model
exp.model_train(model=XGB2Classifier(), name='XGB2')

# Train Model with monotonicity constraints on PAY_1
exp.model_train(model=XGB2Classifier(mono_increasing_list=("PAY_1", )), name="Mono-XGB2")

#%%
# Evaluate predictive performance of XGB2
exp.model_diagnose(model='XGB2', show='accuracy_table')

#%%
# Evaluate predictive performance of Mono-XGB2
exp.model_diagnose(model='Mono-XGB2', show='accuracy_table')

#%%
# Global effect plot for PAY_1 of XGB2
exp.model_interpret(model='XGB2', show="global_effect_plot", uni_feature="PAY_1", original_scale=True, figsize=(5, 4))

#%%
# Global effect plot for PAY_1 of Mono-XGB2
exp.model_interpret(model='Mono-XGB2', show="global_effect_plot", uni_feature="PAY_1", original_scale=True, figsize=(5, 4))

#%%
# Effect importance
exp.model_interpret(model='Mono-XGB2', show="global_ei", figsize=(5, 4))

#%%
# Feature importance
exp.model_interpret(model='Mono-XGB2', show="global_fi", figsize=(5, 4))

#%%
# Local interpretation by effect
exp.model_interpret(model='Mono-XGB2', show="local_ei", sample_id=0, original_scale=True, figsize=(5, 4))
#%%
# Local interpretation by feature
exp.model_interpret(model='Mono-XGB2', show="local_fi", sample_id=0, original_scale=True, figsize=(5, 4))
