# -*- coding: utf-8 -*-
"""
Fairness Comparison
========================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import GLMClassifier, ExplainableBoostingClassifier

exp = Experiment()
exp.data_loader("SimuCredit", silent=True)
exp.data_summary(feature_exclude=["Race", "Gender"], silent=True)
exp.data_prepare(target="Approved", task_type="classification", silent=True)

#%%
# Train Model
exp.model_train(GLMClassifier(), name="GLM")
exp.model_train(ExplainableBoostingClassifier(), name="EBM")

# %%
# Fairness Metric
metrics_result = exp.model_fairness_compare(models=["GLM", "EBM"],
                                            show="metrics",
                                            metric="AIR",
                                            group_category=["Race", "Gender"],
                                            reference_group=[1., 1.],
                                            protected_group=[0., 0.],
                                            favorable_threshold=0.5,
                                            return_data=True,
                                            figsize=(6, 4))
metrics_result.data

# %%
# Fairness Segmented
segmented_result = exp.model_fairness_compare(models=["GLM", "EBM"],
                                              show="segmented",
                                              metric="AIR",
                                              segment_feature="Balance",
                                              group_category=["Race", "Gender"],
                                              reference_group=[1., 1.],
                                              protected_group=[0., 0.],
                                              favorable_threshold=0.5,
                                              segment_bins=5,
                                              return_data=True,
                                              figsize=(8, 4))
segmented_result.data
