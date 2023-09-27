# -*- coding: utf-8 -*-
"""
Data Quality Check
=====================================

Data quality analysis result using the BikeSharing dataset as example.
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment

exp = Experiment()
exp.data_loader(data="BikeSharing", silent=True)
exp.data_prepare(target="cnt", task_type="regression", silent=True)

# %%
# Data quality check for score distribution plot 
from piml.data.outlier_detection import PCA, CBLOF
exp.data_quality_check(method=PCA(), show='score_distribution', threshold=0.999, figsize=(5, 4))

# %%
# Data quality check for score distribution plot 
exp.data_quality_check(method=PCA(), show='marginal_outlier_distribution', threshold=0.999, figsize=(5, 4))

# %%
# Feature selections using distance correlation strategy 
exp.data_quality_check(method=[PCA(), CBLOF()], show='tsne_comparison', threshold=[0.999, 0.999], figsize=(5, 4))
