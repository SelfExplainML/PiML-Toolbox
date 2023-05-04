# -*- coding: utf-8 -*-
"""
Data Load
=====================================

Data loader example
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
import pandas as pd

exp = Experiment()
exp.data_loader(data="CoCircles")

#load bikesharing dataset from internet.
data_pandas = pd.read_csv('https://github.com/SelfExplainML/PiML-Toolbox/blob/main/datasets/BikeSharing.csv?raw=true')
exp.data_loader(data=data_pandas)