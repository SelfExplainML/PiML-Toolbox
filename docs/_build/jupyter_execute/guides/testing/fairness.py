#!/usr/bin/env python
# coding: utf-8

# In[ ]:


metrics_result = exp.model_fairness(model="XGB2_monotonic", show="metrics", metric="AIR",
                group_category=["Race", "Gender"], reference_group=[1., 1.],
                protected_group=[0., 0.], favorable_threshold=0.5,
                figsize=(6, 4), return_data=True)


# In[ ]:


segmented_result = exp.model_fairness(model="XGB2_monotonic", show="segmented", metric="AIR",
                                       segment_feature="Balance", group_category=["Race","Gender"],
                                       reference_group=[1., 1.], protected_group=[0., 0.],
                                       segment_bins=5, favorable_threshold=0.5,
                                       return_data=True, figsize=(8, 4))


# In[ ]:


binning_dict = {"Balance": {"type": "quantile", "value": [1, 5]},
                "Mortgage": {"type": "uniform", "value": [1, 5]},
                "Amount Past Due": {"type": "custom", "value": (0, 100)}}
binning_result = exp.model_fairness(model="XGB2_monotonic", show="binning", metric="AIR",
                                     group_category=["Race","Gender"],
                                     reference_group=[1., 1.], protected_group=[0., 0.],
                                     favorable_threshold=0.5, performance_metric="F1",
                                     binning_dict=binning_dict, return_data=True, figsize=(8,4))


# In[ ]:


thresholding_result = exp.model_fairness(model="XGB2_monotonic", show="thresholding", metric="AIR",
                                         group_category=["Race","Gender"],
                                         reference_group=[1., 1.], protected_group=[0., 0.],
                                         favorable_threshold=0.32, performance_metric="ACC",
                                         return_data=True)

