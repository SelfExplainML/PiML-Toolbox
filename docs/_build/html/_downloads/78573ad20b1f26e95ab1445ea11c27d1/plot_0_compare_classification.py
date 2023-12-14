# -*- coding: utf-8 -*-
"""
Model Comparison: Classification
========================================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import TreeClassifier
from piml.models import FIGSClassifier
from piml.models import XGB2Classifier

exp = Experiment()
exp.data_loader(data="TaiwanCredit", silent=True)
exp.data_summary(feature_exclude=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"], silent=True)
exp.data_prepare(target="FlagDefault", task_type="classification", silent=True)

#%%
# Train Model
exp.model_train(TreeClassifier(), name="Tree")
exp.model_train(FIGSClassifier(), name="FIGS")
exp.model_train(XGB2Classifier(), name="XGB2")

#%%
# Accuracy comparison with ACC
exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="accuracy_plot", metric="ACC", figsize=(5, 4))

#%%
# Accuracy comparison with AUC
exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="accuracy_plot", metric="AUC", figsize=(5, 4))

#%%
# Accuracy comparison with F1
exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="accuracy_plot", metric="F1", figsize=(5, 4))

#%%
# Overfit comparison with ACC metric
exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="overfit",
                  slice_method="histogram", slice_feature="PAY_1", 
                  bins=10, metric="ACC", original_scale=True, figsize=(5, 4))
#%%
# Overfit comparison with AUC metric
exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="overfit",
                  slice_method="histogram", slice_feature="PAY_1", 
                  metric="AUC", original_scale=True, figsize=(5, 4))

#%%
# Reliability bandwidth comparison 
exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="reliability_bandwidth", figsize=(5, 4))
#%%
# Reliability diagram comparison 
exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="reliability_perf", bins=10, figsize=(5, 4))

#%%
# Robustness comparison with default settings
exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="robustness_perf", figsize=(5, 4))

#%%
# Robustness comparison on worst alpha-percent samples
exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="robustness_perf_worst", alpha=0.3, figsize=(5, 4))

#%%
# Resilience comparison with worst-sample scenario
exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="resilience_perf",
                  resilience_method="worst-sample", immu_feature=None, metric="AUC", figsize=(5, 4))

#%%
# Resilience distance comparison with worst-sample scenario
exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="resilience_distance",
                  resilience_method="worst-sample", metric="AUC", alpha=0.3, figsize=(5, 4))
