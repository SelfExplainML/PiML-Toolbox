# -*- coding: utf-8 -*-
"""
FIGS: Regression
========================

"""
#%%
from piml import Experiment
from piml.models import FIGSRegressor
exp = Experiment()
exp.data_loader(data='CaliforniaHousing_trim2')
exp.data_prepare(target='MedHouseVal', task_type='Regression', test_ratio=0.2, random_state=1)
   
#%% Model training
exp.model_train(model=FIGSRegressor(max_iter=100, max_depth=4), name='FIGS')

#%%
# Global interpretation for the first tree.
exp.model_interpret(model='FIGS', show="tree_global", root=0, tree_idx=0,
                         depth=3, original_scale=True, figsize=(20, 6))

#%%
# Global interpretation for the second tree.
exp.model_interpret(model='FIGS', show="tree_global", root=0, tree_idx=1,
                         depth=3, original_scale=True, figsize=(20, 6))

#%%
# Global interpretation for the second tree.
exp.model_interpret(model='FIGS', show="tree_local", sample_id=0, tree_idx=0, root=0,
                        depth=3, original_scale=True, figsize=(20, 6))