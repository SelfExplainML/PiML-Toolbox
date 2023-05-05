# -*- coding: utf-8 -*-
"""
Data Prepare
=====================================

Display data prepare result using the BikeSharing dataset as example.
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment

exp = Experiment()
exp.data_loader(data="BikeSharing", silent=True)

exp.data_prepare(target='cnt', task_type='Regression', sample_weight=None,
                       split_method='random', test_ratio=0.2, random_state=0)

import numpy as np
custom_train_idx = np.arange(0,16000)
custom_test_idx = np.arange(16000, 17379)
exp.data_prepare(train_idx=custom_train_idx, test_idx=custom_test_idx)