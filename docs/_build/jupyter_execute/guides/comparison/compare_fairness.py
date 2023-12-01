#!/usr/bin/env python
# coding: utf-8

# In[ ]:


metrics_result = exp.model_fairness_compare(models=["GLM", "EBM"], show="metrics", metric="AIR",
                                            group_category=["Race", "Gender"],
                                            reference_group=[1., 1.], protected_group=[0., 0.],
                                            favorable_threshold=0.5,
                                            figsize=(6, 4), return_data=True)


# In[ ]:


segmented_result = exp.model_fairness_compare(models=["GLM", "EBM"], show="segmented", metric="AIR",
                                              segment_feature="Balance",
                                              group_category=["Race", "Gender"],
                                              reference_group=[1., 1.], protected_group=[0., 0.],
                                              segment_bins=5, favorable_threshold=0.5,
                                              return_data=True, figsize=(8, 4))

