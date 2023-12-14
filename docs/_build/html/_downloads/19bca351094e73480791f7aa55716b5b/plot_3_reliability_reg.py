# -*- coding: utf-8 -*-
"""
Reliability: Regression
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
exp.model_train(XGB2Regressor(), name="XGB2")

#%%
#Show empirical coverage and average bandwidth for regression tasks
results = exp.model_diagnose(model="XGB2", show="reliability_table", alpha=0.1, return_data=True)
results.data

#%%
#Calculate distribution shift distance of each feature between reliable and un-reliable data
exp.model_diagnose(model="XGB2", show="reliability_distance", alpha=0.1,
                   threshold=1.1, distance_metric="PSI", figsize=(5, 4))

#%%
#Plot the histogram of bandwidth against a given feature
exp.model_diagnose(model="XGB2", show="reliability_marginal", alpha=0.1,
                   show_feature="hr", bins=10, threshold=1.1,
                   original_scale=True, figsize=(5, 4))
