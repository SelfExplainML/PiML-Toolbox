# -*- coding: utf-8 -*-
"""
FIGS Regression (California Housing)
===========================================
"""
#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import FIGSRegressor

exp = Experiment()
exp.data_loader(data="CaliforniaHousing_trim2", silent=True)
exp.data_prepare(target="MedHouseVal", task_type="regression", silent=True)
   
#%% 
# Train Model
exp.model_train(model=FIGSRegressor(max_iter=100, max_depth=4), name="FIGS")

#%%
# Evaluate predictive performance
exp.model_diagnose(model="FIGS", show="accuracy_table")

#%%
# Global interpretation for the splits heatmap
exp.model_interpret(model="FIGS", show="figs_heatmap", tree_idx=0, figsize=(12, 4))

#%%
# Global interpretation for the first tree
exp.model_interpret(model="FIGS", show="tree_global", root=0, tree_idx=0,
                    depth=3, original_scale=True, figsize=(16, 10))
#%%
# Global interpretation for the second tree
exp.model_interpret(model="FIGS", show="tree_global", root=0, tree_idx=1,
                    depth=3, original_scale=True, figsize=(16, 10))

#%%
# Local interpretation for the first tree
exp.model_interpret(model="FIGS", show="tree_local", sample_id=0, tree_idx=0,
                    original_scale=True, figsize=(16, 10))

#%%
# Local interpretation for the second tree
exp.model_interpret(model="FIGS", show="tree_local", sample_id=0, tree_idx=1, 
                    original_scale=True, figsize=(16, 10))
