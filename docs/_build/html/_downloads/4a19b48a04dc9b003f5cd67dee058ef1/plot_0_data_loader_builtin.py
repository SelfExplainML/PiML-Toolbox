# -*- coding: utf-8 -*-
"""
Data Load (Built-in Dataset)
=====================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment

exp = Experiment()
exp.data_loader(data="CoCircles")
