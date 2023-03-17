# -*- coding: utf-8 -*-
"""
FIGS: Regression
========================

"""
#%%
from piml import Experiment
from piml.models import FIGSClassifier
exp = Experiment()
exp.data_loader(data='TaiwanCredit')
exp.data_summary(feature_exclude=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"], feature_type={})
exp.data_prepare(target='FlagDefault', task_type='Classification', test_ratio=0.2, random_state=0)
   
#%% Model training
exp.model_train(model=FIGSClassifier(max_iter=100, max_depth=4), name='FIGS')

#%%
# Global interpretation for the first tree.
exp.model_interpret(model='FIGS', show="tree_global", tree_idx=0, root=0, 
                    depth=3, original_scale=True, figsize=(20, 6))

#%%
# Global interpretation for the second tree.
exp.model_interpret(model='FIGS', show="tree_global", tree_idx=1, root=0, 
                    depth=3, original_scale=True, figsize=(20, 6))

#%%
# Global interpretation for the second tree.
exp.model_interpret(model='FIGS', show="tree_local", sample_id=0, tree_idx=0,
                    original_scale=True, figsize=(20, 6))