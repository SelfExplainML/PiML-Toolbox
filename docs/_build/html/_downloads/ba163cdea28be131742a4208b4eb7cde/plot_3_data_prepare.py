# -*- coding: utf-8 -*-
"""
Data Preparation
=====================================

Display data prepare result using the BikeSharing dataset as example.
"""

#%%
# Experiment initialization and data preparation
import numpy as np
from piml import Experiment

exp = Experiment()
exp.data_loader(data="BikeSharing", silent=True)

#%%
# Random split
exp.data_prepare(target='cnt', task_type='regression', sample_weight=None,
                 split_method='random', test_ratio=0.2, random_state=0)

#%%
# Outer-sample-based split
exp.data_prepare(target='cnt', task_type='regression', sample_weight=None,
                 split_method='outer-sample', test_ratio=0.2, random_state=0)

#%%
# KMeans-based split
exp.data_prepare(target='cnt', task_type='regression', sample_weight=None,
                 split_method='kmeans', test_ratio=[0.0, 1.0, 0.0], random_state=0)

#%%
# Custom split
custom_train_idx = np.arange(0, 16000)
custom_test_idx = np.arange(16000, 17379)
exp.data_prepare(target='cnt', task_type='regression', sample_weight=None, 
                 train_idx=custom_train_idx, test_idx=custom_test_idx)