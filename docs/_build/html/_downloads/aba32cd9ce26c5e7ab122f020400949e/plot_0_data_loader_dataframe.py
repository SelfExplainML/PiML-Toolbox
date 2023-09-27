# -*- coding: utf-8 -*-
"""
Data Load (Pandas DataFrame)
=====================================
"""

#%%
# Experiment initialization and data preparation
import pandas as pd
from piml import Experiment

exp = Experiment()
data = pd.read_csv('https://github.com/SelfExplainML/PiML-Toolbox/blob/main/datasets/BikeSharing.csv?raw=true')
exp.data_loader(data=data)
