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