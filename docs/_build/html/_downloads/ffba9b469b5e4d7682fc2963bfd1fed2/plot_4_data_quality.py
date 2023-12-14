# -*- coding: utf-8 -*-
"""
Data Quality Check
=====================================
Data quality analysis result using the BikeSharing dataset as example
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.data.outlier_detection import (PCA, CBLOF, IsolationForest, KMeansTree, 
                                         OneClassSVM, KNN, HBOS, ECOD)

exp = Experiment()
exp.data_loader(data="BikeSharing", silent=True)
exp.data_summary(feature_exclude=["yr", "mnth", "temp"], silent=True)
exp.data_prepare(target="cnt", task_type="regression", silent=True)

# %%
# Data integrity check for each column 
res = exp.data_quality(show='integrity_single_column_check', return_data=True)
res.data

# %%
# Data integrity check for duplicated samples 
res = exp.data_quality(show='integrity_duplicated_samples', return_data=True)
res.data

# %%
# Data integrity check for correlated features 
res = exp.data_quality(show='integrity_highly_correlated_features', return_data=True)
res.data

# %%
# Data quality check for score distribution plot 
exp.data_quality(method=PCA(), show='od_score_distribution', threshold=0.999, figsize=(5, 4))

# %%
# Data quality check for score distribution plot 
exp.data_quality(method=CBLOF(), show='od_score_distribution', threshold=0.999, figsize=(5, 4))

# %%
# Data quality check for score distribution plot
exp.data_quality(method=IsolationForest(), show='od_score_distribution', threshold=0.999, figsize=(5, 4))

# %%
# Data quality check for score distribution plot
exp.data_quality(method=KMeansTree(), show='od_score_distribution', threshold=0.999, figsize=(5, 4))

# %%
# Data quality check for score distribution plot 
exp.data_quality(method=KNN(), show='od_score_distribution', threshold=0.999, figsize=(5, 4))

# %%
# Data quality check for score distribution plot 
exp.data_quality(method=HBOS(), show='od_score_distribution', threshold=0.999, figsize=(5, 4))

# %%
# Data quality check for score distribution plot 
exp.data_quality(method=ECOD(), show='od_score_distribution', threshold=0.999, figsize=(5, 4))

# %%
# Data quality check for score distribution plot 
exp.data_quality(method=OneClassSVM(), show='od_score_distribution', threshold=0.999, figsize=(5, 4))

# %%
# Data quality check for score distribution plot 
exp.data_quality(method=PCA(), show='od_marginal_outlier_distribution',
                 threshold=0.999, figsize=(5, 4))

# %%
# Compare different outlier detection algorithms  
exp.data_quality(method=[PCA(), CBLOF()], show='od_tsne_comparison',
                 threshold=[0.999, 0.999], figsize=(5, 4))

# %%
# Select a method and threshold and apply the outlier removal (you can also specify train, test, or all data)
exp.data_quality(method=CBLOF(), show='od_score_distribution', dataset="train",
                 threshold=0.999, remove_outliers=True, figsize=(5, 4))

# %%
# Compare the train and test data energy distance.
exp.data_quality(show='drift_test_info')

# %%
# Compare the train and test marginal data drift feature-by-feature. 
exp.data_quality(show='drift_test_distance', figsize=(5, 4))

# %%
# Compare the train and test marginal data drift of a given feature. 
exp.data_quality(show="drift_test_distance", distance_metric="PSI", psi_buckets='quantile',
                 show_feature="atemp", figsize=(5, 4))
