# -*- coding: utf-8 -*-
"""
H-statistics 
=====================================

"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import XGB2Regressor

exp = Experiment()
exp.data_loader(data="Friedman", silent=True)
exp.data_summary(silent=True)
exp.data_prepare(target="target", task_type="regression", silent=True)

#%%
# Train Model
exp.model_train(model=XGB2Regressor(n_estimators=100), name="XGB2")

#%%
# Run H-statistics with 2000 subsampled data and grid size of 5
res = exp.model_explain(model="XGB2", show="hstats", sample_size=2000, grid_size=5,
                        return_data=True, figsize=(5, 4))
res.data
