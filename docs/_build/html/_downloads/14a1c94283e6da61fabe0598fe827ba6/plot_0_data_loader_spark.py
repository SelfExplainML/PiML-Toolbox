# -*- coding: utf-8 -*-
"""
Data Load (Spark)
=====================================
"""

#%%
# Generate a parquet data for demonstration
import numpy as np
import pandas as pd

np.random.seed(0)
original_df = pd.DataFrame(
    np.hstack([np.random.randint(2, size=(100000, 1)), np.random.uniform(-1, 1, size=(100000, 10))]),
    columns=["Y"] + ["X" + str(i) for i in range(10)]
   )
original_df.to_parquet('myfile.parquet')

#%%
# Experiment initialization
from piml import Experiment
exp = Experiment()

#%%
# Data loading with 10000 samples (purly randomly)
exp.data_loader(data="./myfile.parquet", spark=True, spark_sample_size=10000, spark_random_state=0)


#%%
# Data loading with 10000 samples (stratified sampling)
exp.data_loader(data="./myfile.parquet", spark=True, spark_sample_size=10000,
                spark_sample_by_feature='Y', spark_random_state=0)


#%%
# Data loading with 10000 samples (stratified sampling with given uneven ratios)
exp.data_loader(data="./myfile.parquet", spark=True, spark_sample_size=10000,
                spark_sample_by_feature='Y', spark_sample_fractions={0.0: 1, 1.0: 5},
                spark_random_state=0)
