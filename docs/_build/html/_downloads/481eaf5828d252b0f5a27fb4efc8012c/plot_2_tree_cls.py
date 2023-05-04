# -*- coding: utf-8 -*-
"""
Tree Classification (TaiwanCredit)
=========================================
"""
#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import TreeClassifier

exp = Experiment()
exp.data_loader(data="TaiwanCredit", silent=True)
exp.data_summary(feature_exclude=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"], silent=True)
exp.data_prepare(target="FlagDefault", task_type="classification", silent=True)

#%% 
# Train Model
exp.model_train(model=TreeClassifier(max_depth=6), name="Tree")

#%%
# Evaluate predictive performance
exp.model_diagnose(model="Tree", show="accuracy_table")

#%%
# Global interpretation starting from the root node
exp.model_interpret(model="Tree", show="tree_global", root=0, depth=3,
                    original_scale=True, figsize=(16, 10))

#%%
# Global interpretation starting from the 10-th node
exp.model_interpret(model="Tree", show="tree_global", root=2, depth=3,
                    original_scale=True, figsize=(16, 10))

#%%
# Local interpretation
exp.model_interpret(model="Tree", show="tree_local", sample_id=0,
                    original_scale=True, figsize=(16, 10))
