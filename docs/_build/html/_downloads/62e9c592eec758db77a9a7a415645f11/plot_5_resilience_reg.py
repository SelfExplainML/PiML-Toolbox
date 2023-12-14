# -*- coding: utf-8 -*-
"""
Resilience - Regression 
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

# %%
# Train model
exp.model_train(model=XGB2Regressor(n_estimators=100), name="XGB2")

#%%
# Resilience performance against worst sample scenario
exp.model_diagnose(model="XGB2", show="resilience_perf", resilience_method="worst-sample", figsize=(5, 4))
#%%
# Resilience performance against outer sample scenario
exp.model_diagnose(model="XGB2", show="resilience_perf", resilience_method="outer-sample", figsize=(5, 4))
#%%
# Resilience performance against worst cluster scenario
exp.model_diagnose(model="XGB2", show="resilience_perf", resilience_method="worst-cluster", figsize=(5, 4))
#%%
# Resilience performance against hard sample scenario
exp.model_diagnose(model="XGB2", show="resilience_perf", resilience_method="hard-sample", figsize=(5, 4))
#%%
# Marginal distributional distance between full sample and worst sample with worst-sample scenario
exp.model_diagnose(model="XGB2", show="resilience_distance", resilience_method="worst-sample", 
                   distance_metric="PSI", alpha=0.3, figsize=(5, 4))
#%%
# Marginal distributional distance between full sample and worst sample with not None immutable features
exp.model_diagnose(model="XGB2", show="resilience_distance", resilience_method="worst-sample", 
                         distance_metric="PSI", immu_feature="hr", alpha=0.3, figsize=(5, 4))
#%%
# Marginal distributional distance between full sample and worst sample with worst-cluster scenario
exp.model_diagnose(model="XGB2", show="resilience_distance", resilience_method="worst-cluster",
                   distance_metric="WD1", n_clusters=10, figsize=(5, 4))
#%%
# Marginal histogram plot for full sample and worst sample
exp.model_diagnose(model="XGB2", show="resilience_shift_histogram", resilience_method="worst-sample",
                   show_feature="hr", original_scale=True, figsize=(5, 4))
#%%
# Marginal density plot for full sample and worst sample
exp.model_diagnose(model="XGB2", show="resilience_shift_density", resilience_method="worst-sample",
                   show_feature="hr", original_scale=True, figsize=(5, 4))
