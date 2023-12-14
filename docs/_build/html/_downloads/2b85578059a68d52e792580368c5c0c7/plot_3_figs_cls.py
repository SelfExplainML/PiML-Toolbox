# -*- coding: utf-8 -*-
"""
FIGS Classification (Taiwan Credit)
===========================================
"""
#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import FIGSClassifier

exp = Experiment()
exp.data_loader(data="TaiwanCredit", silent=True)
exp.data_summary(feature_exclude=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"], silent=True)
exp.data_prepare(target="FlagDefault", task_type="classification", silent=True)
   
#%% 
# Train Model
exp.model_train(model=FIGSClassifier(max_iter=100, max_depth=4), name="FIGS")

#%%
# Evaluate predictive performance
exp.model_diagnose(model="FIGS", show="accuracy_table")

#%%
# Global interpretation for the splits heatmap
exp.model_interpret(model="FIGS", show="figs_heatmap", tree_idx=0, figsize=(12, 4))

#%%
# Global interpretation for the first tree
exp.model_interpret(model="FIGS", show="tree_global", tree_idx=0, root=0, 
                    depth=3, original_scale=True, figsize=(16, 10))

#%%
# Global interpretation for the second tree
exp.model_interpret(model="FIGS", show="tree_global", tree_idx=1, root=0, 
                    depth=3, original_scale=True, figsize=(16, 10))

#%%
# Local interpretation for the first tree
exp.model_interpret(model="FIGS", show="tree_local", sample_id=0, tree_idx=0,
                    original_scale=True, figsize=(16, 10))

#%%
# Local interpretation for the second tree
exp.model_interpret(model="FIGS", show="tree_local", sample_id=0, tree_idx=1,
                    original_scale=True, figsize=(16, 10))
