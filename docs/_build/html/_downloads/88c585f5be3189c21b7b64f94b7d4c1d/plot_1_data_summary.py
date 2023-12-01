# -*- coding: utf-8 -*-
"""
Data Summary
=====================================

Showing data summary result using the BikeSharing dataset as example.
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment

exp = Experiment(highcode_only=True)
exp.data_loader(data="BikeSharing", silent=True)

#%%
# Feature removal
exp.data_summary(feature_exclude=["yr", "mnth", "temp"], silent=True)

#%%
# Change feature types
exp.data_summary(feature_exclude=["yr", "mnth", "temp"], feature_type={"weekday": "categorical"}, silent=True)
