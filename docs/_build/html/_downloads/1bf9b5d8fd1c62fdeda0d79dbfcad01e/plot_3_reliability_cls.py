# -*- coding: utf-8 -*-
"""
Reliability: Classification
=====================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import GAMClassifier

exp = Experiment()
exp.data_loader(data="TaiwanCredit", silent=True)
exp.data_summary(feature_exclude=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"], silent=True)
exp.data_prepare(target="FlagDefault", task_type="classification", silent=True)

#%%
# Train Model
exp.model_train(GAMClassifier(), name="GAM")

#%%
#Calculate distribution shift distance of each feature between reliable and un-reliable data
exp.model_diagnose(model="GAM", show="reliability_distance",
                   threshold=1.1, distance_metric="PSI", figsize=(5, 4))
#%%
#Plot the histogram of bandwidth against a given feature
exp.model_diagnose(model="GAM", show="reliability_marginal",
                   show_feature="PAY_1", bins=10, threshold=1.1, 
                   original_scale=True, figsize=(5, 4))
#%%
#Show the calibrated predicted probability vs. original predicted probability
exp.model_diagnose(model="GAM", show="reliability_calibration", figsize=(5, 4))
#%%
#Reliability diagram
exp.model_diagnose(model="GAM", show="reliability_perf", figsize=(5, 4))
#%%
#Brier Score
exp.model_diagnose(model="GAM", show="reliability_table")
