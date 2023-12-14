# -*- coding: utf-8 -*-
"""
Overfit: Classification
=====================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import XGB2Classifier

#%%
#Load data and train models.
exp = Experiment()
exp.data_loader(data="TaiwanCredit", silent=True)
exp.data_summary(feature_exclude=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"], silent=True)
exp.data_prepare(target="FlagDefault", task_type="classification", silent=True)

#%%
# Train Model
exp.model_train(XGB2Classifier(), name="XGB2")

#%%
# Histogram-based overfit test for a single feature
results = exp.model_diagnose(model="XGB2", show="overfit", slice_method="histogram", 
                             slice_features=["BILL_AMT1"], threshold=1.05, min_samples=20,
                             original_scale=True, return_data=True, figsize=(5, 4))
results.data
#%%
# Histogram-based overfit test for two features
results = exp.model_diagnose(model="XGB2", show="overfit", slice_method="histogram", 
                             slice_features=["PAY_1", "BILL_AMT1"], threshold=1.05, min_samples=20,
                             original_scale=True, return_data=True, figsize=(5, 4))
results.data

#%%
# Histogram-based overfit test for a single feature on test set
results = exp.model_diagnose(model="XGB2", show="overfit", slice_method="histogram", 
                             slice_features=["BILL_AMT1"], threshold=1.05, min_samples=20,
                             use_test=True, original_scale=True, return_data=True, figsize=(5, 4))
results.data
