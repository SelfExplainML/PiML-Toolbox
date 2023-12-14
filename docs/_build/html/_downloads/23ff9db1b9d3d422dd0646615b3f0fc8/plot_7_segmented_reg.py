# -*- coding: utf-8 -*-
"""
Segmented Diagnose (Regression)
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
exp.model_train(XGB2Regressor(), name="XGB2")

#%%
# Summary of all segments (top 10 with the worst performance)
result = exp.segmented_diagnose(model='XGB2', show='segment_table',
                                segment_method='uniform', segment_bins=5, return_data=True)
result.data.head(10)

#%%
# Summary of all segments of a given feature (top 10 with the worst performance)
result = exp.segmented_diagnose(model="XGB2", show="segment_table",
                                segment_method="uniform", segment_feature="hr", segment_bins=5, return_data=True)
result.data

#%%
# Residual analysis of the samples in that segment
exp.segmented_diagnose(model="XGB2", show="accuracy_residual", 
                       segment_method="uniform", segment_feature="hr", segment_bins=5, segment_id=0,
                       show_feature="atemp", figsize=(5, 4))

#%%
# Weakspot analysis of the samples in that segment
exp.segmented_diagnose(model="XGB2", show="weakspot", 
                       segment_method="uniform", segment_feature="hr", segment_bins=5, segment_id=0,
                       slice_features=["atemp"], metric="MSE", figsize=(5, 4))

#%%
# Distributional distance comparison between the specificed segment and the remaining (feature-by-feature)
res = exp.segmented_diagnose(model="XGB2", show="distribution_shift",
                             segment_method="uniform", segment_feature="hr", segment_bins=5, segment_id=0,
                             figsize=(5, 4), return_data=True)

#%%
# Distributional distance comparison between the specificed segment and the remaining (density of one selected feature)
res = exp.segmented_diagnose(model="XGB2", show="distribution_shift",
                             segment_method="uniform", segment_feature="hr", segment_bins=5,
                             segment_id=0, show_feature="hum", figsize=(5, 4), return_data=True)
