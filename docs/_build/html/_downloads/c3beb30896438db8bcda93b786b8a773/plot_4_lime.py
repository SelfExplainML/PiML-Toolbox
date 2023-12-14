# -*- coding: utf-8 -*-
"""
Local Interpretable Model-Agnostic Explanation
========================================================

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
# Plot LIME without centering
exp.model_explain(model="XGB2", show="lime", sample_id=0, centered=False, original_scale=True, figsize=(5, 4))
#%%
# Plot LIME with centering
exp.model_explain(model="XGB2", show="lime", sample_id=0, centered=True, original_scale=True, figsize=(5, 4))
