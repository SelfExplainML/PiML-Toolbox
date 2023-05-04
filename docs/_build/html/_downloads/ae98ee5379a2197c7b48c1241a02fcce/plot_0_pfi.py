# -*- coding: utf-8 -*-
"""
Permutation Feature Importance
=====================================

"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import XGB2Regressor

exp = Experiment()
exp.data_loader(data="BikeSharing", silent=True)
exp.data_summary(feature_exclude=["yr", "mnth", "temp"], silent=True)
exp.data_prepare(target="cnt", task_type="regression", silent=True)

#%%
# Train Model
exp.model_train(model=XGB2Regressor(), name="XGB2")

#%%
# PFI Plot
exp.model_explain(model="XGB2", show="pfi", n_repeats=10, figsize=(5, 4))
