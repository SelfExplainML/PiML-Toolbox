# -*- coding: utf-8 -*-
"""
WeakSpot: Classification    
=====================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import XGB2Classifier

exp = Experiment()
exp.data_loader(data="TaiwanCredit", silent=True)
exp.data_summary(feature_exclude=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"], silent=True)
exp.data_prepare(target="FlagDefault", task_type="classification", silent=True)

#%%
#Train Model
exp.model_train(XGB2Classifier(), name="XGB2")

#%%
# Histogram-based weakspot for a single feature
results = exp.model_diagnose(model="XGB2", show="weakspot", slice_method="histogram", 
                             slice_features=["PAY_1"], threshold=1.1, min_samples=100,
                             return_data=True, figsize=(5, 4))
results.data

#%%
# Histogram-based weakspot for two features
results = exp.model_diagnose(model="XGB2", show="weakspot", slice_method="histogram", 
                             slice_features=["PAY_1", "PAY_2"], threshold=1.1, min_samples=100,
                             return_data=True, figsize=(5, 4))
results.data

#%%
# Histogram-based weakspot for a single feature on test set
results = exp.model_diagnose(model="XGB2", show="weakspot", slice_method="histogram", 
                             slice_features=["PAY_1"], threshold=1.1, min_samples=100,
                             use_test=True, return_data=True, figsize=(5, 4))
results.data
#%%
# Histogram-based weakspot for a single feature using AUC metric
results = exp.model_diagnose(model="XGB2", show="weakspot", slice_method="histogram", 
                             slice_features=["PAY_1"], threshold=1.1, min_samples=100,
                             metric="AUC", return_data=True, figsize=(5, 4))
results.data

#%%
# Tree-based weakspot for a single feature using ACC metric
results = exp.model_diagnose(model="XGB2", show="weakspot", slice_method="tree", 
                             slice_features=["PAY_1"], threshold=1.1, min_samples=100,
                             metric="ACC", return_data=True, figsize=(5, 4))
results.data

