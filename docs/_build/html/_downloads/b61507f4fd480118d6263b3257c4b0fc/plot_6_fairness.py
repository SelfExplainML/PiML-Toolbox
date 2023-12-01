# -*- coding: utf-8 -*-
"""
Fairness Test: XGB2
=====================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import XGB2Classifier

exp = Experiment()
exp.data_loader("SimuCredit", silent=True)
exp.data_summary(feature_exclude=["Race", "Gender"], silent=True)
exp.data_prepare(target="Approved", task_type="classification", silent=True)

#%%
# Train Model
exp.model_train(XGB2Classifier(max_depth=2,
                               n_estimators=100, 
                               mono_increasing_list=["Mortgage", "Balance"],
                               mono_decreasing_list=["Amount Past Due", "Utilization", "Delinquency",
                                                     "Credit Inquiry", "Open Trade"]),
                name="XGB2_monotonic")

# %%
# Fairness Metric
metrics_result = exp.model_fairness(model="XGB2_monotonic",
                                    show="metrics",
                                    metric="AIR",
                                    group_category=["Race", "Gender"],
                                    reference_group=[1., 1.],
                                    protected_group=[0., 0.],
                                    favorable_threshold=0.5,
                                    performance_metric="ACC",
                                    return_data=True,
                                    figsize=(5, 4))
metrics_result.data

#%%
# Fairness Segmented
segmented_result = exp.model_fairness(model="XGB2_monotonic",
                                      show="segmented",
                                      metric="AIR",
                                      segment_feature="Balance",
                                      group_category=["Race", "Gender"],
                                      reference_group=[1., 1.],
                                      protected_group=[0., 0.],
                                      favorable_threshold=0.5,
                                      segment_bins=5,
                                      return_data=True,
                                      figsize=(10, 4))
segmented_result.data

#%%
# Fairness Binning
binning_result = exp.model_fairness(model="XGB2_monotonic", show="binning",
                                    metric="AIR",
                                    group_category=["Race", "Gender"],
                                    reference_group=[1., 1.],
                                    protected_group=[0., 0.],
                                    favorable_threshold=0.5,
                                    performance_metric="F1",
                                    binning_dict={"Balance": {"type": "quantile", "value": [1, 5]},
                                                  "Mortgage": {"type": "uniform", "value": [1, 5]},
                                                  "Amount Past Due": {"type": "custom", "value": (0, 100)}},
                                    return_data=True,
                                    figsize=(10, 4))
binning_result.data

#%%
# Fairness Thresholding
thresholding_result = exp.model_fairness(model="XGB2_monotonic",
                                         show="thresholding",
                                         metric="AIR", 
                                         group_category=["Race", "Gender"],
                                         reference_group=[1., 1.],
                                         protected_group=[0., 0.],
                                         favorable_threshold=0.32,
                                         performance_metric="ACC",
                                         return_data=True,
                                         figsize=(10, 4))
thresholding_result.data
