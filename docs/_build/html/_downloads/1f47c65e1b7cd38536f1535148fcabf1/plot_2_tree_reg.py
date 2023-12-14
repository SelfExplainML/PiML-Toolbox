# -*- coding: utf-8 -*-
"""
Tree Regression (California Housing)
=========================================
"""
#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import TreeRegressor

exp = Experiment()
exp.data_loader(data="CaliforniaHousing_trim2", silent=True)
exp.data_prepare(target="MedHouseVal", task_type="regression", silent=True)

#%% 
# Train Model
exp.model_train(model=TreeRegressor(max_depth=6), name="Tree")

#%%
# Evaluate predictive performance
exp.model_diagnose(model="Tree", show="accuracy_table")

#%%
# Global interpretation starting from the root node
exp.model_interpret(model="Tree", show="tree_global", root=0, depth=3,
                    original_scale=True, figsize=(16, 10))

#%%
# Global interpretation starting from the second node
exp.model_interpret(model="Tree", show="tree_global", root=2, depth=3, 
                    original_scale=True, figsize=(16, 10))

#%%
# Local interpretation
exp.model_interpret(model="Tree", show="tree_local", sample_id=0, 
                    original_scale=True, figsize=(16, 10))
