# -*- coding: utf-8 -*-
"""
Two Sample Test
=====================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment

exp = Experiment()
exp.data_loader(data="BikeSharing", silent=True)
exp.data_prepare(target="cnt", task_type="regression", silent=True)

# %%
# Distributional shift distance with PSI
exp.twosample_test(metric="PSI", psi_buckets='uniform', figsize=(5, 4))

# %%
# Distributional shift distance with WD1
exp.twosample_test(metric="WD1", figsize=(5, 4))

# %%
# Distributional shift distance with KS
exp.twosample_test(metric="KS", figsize=(5, 4))

# %%
# Distributional shift plot for a single feature
exp.twosample_test(metric="PSI", psi_buckets='quantile', feature="atemp", figsize=(5, 4))
