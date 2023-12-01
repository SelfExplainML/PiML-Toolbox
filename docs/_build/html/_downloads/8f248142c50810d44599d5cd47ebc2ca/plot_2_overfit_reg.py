# -*- coding: utf-8 -*-
"""

Overfit: Regression
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
# Histogram-based overfit test for a single feature
results = exp.model_diagnose(model="XGB2", show="overfit", slice_method="histogram", 
                             slice_features=["hr"], threshold=1.05, min_samples=100,
                             return_data=True, figsize=(5, 4))
results.data
#%%
# Histogram-based overfit test for two features
results = exp.model_diagnose(model="XGB2", show="overfit", slice_method="histogram", 
                             slice_features=["hr", "atemp"], threshold=1.05, min_samples=100,
                             return_data=True, figsize=(5, 4))
results.data
#%%
# Histogram-based overfit test for a single feature using MAE metric
results = exp.model_diagnose(model="XGB2", show="overfit", slice_method="histogram", 
                             slice_features=["atemp"], threshold=1.05, min_samples=100,
                             metric="MAE", return_data=True, figsize=(5, 4))
results.data
