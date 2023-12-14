# -*- coding: utf-8 -*-
"""
Robustness: Classification
=====================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import ReluDNNClassifier

exp = Experiment()
exp.data_loader(data="TaiwanCredit", silent=True)
exp.data_summary(feature_exclude=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"], silent=True)
exp.data_prepare(target="FlagDefault", task_type="classification", silent=True)

#%%
# Train Model
exp.model_train(ReluDNNClassifier(), name="ReluDNN")

#%%
# Robustness test with default settings
exp.model_diagnose(model="ReluDNN", show="robustness_perf", figsize=(6, 4))
#%%
# Robustness test with custom perturbation features
exp.model_diagnose(model="ReluDNN", show="robustness_perf",
                   perturb_features=["BILL_AMT1", "BILL_AMT2", "BILL_AMT3"], figsize=(6, 4))

#%%
# Robustness test with custom perturbation size
exp.model_diagnose(model="ReluDNN", show="robustness_perf", perturb_size=0.2, figsize=(6, 4))

#%%
# Robustness test with custom perturbation method
exp.model_diagnose(model="ReluDNN", show="robustness_perf", perturb_method="quantile", figsize=(6, 4))

#%%
# Robustness test with custom metrics
exp.model_diagnose(model="ReluDNN", show="robustness_perf", metric="AUC", figsize=(6, 4))

#%%
# Robustness test on worst alpha-percent samples
exp.model_diagnose(model="ReluDNN", show="robustness_perf_worst", alpha=0.3, figsize=(6, 4))
