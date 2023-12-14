#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="accuracy_plot",
                  metric="MSE", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="accuracy_plot",
                  metric="MAE", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="accuracy_plot",
                  metric="R2", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="overfit",
                  metricmetric="MSE", slice_method="histogram", bins=10,
                  slice_feature="hr", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="overfit",
                  slice_method="histogram", slice_feature="hr",
                  metric="MAE", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="reliability_bandwidth",
                  alpha=0.1, figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="reliability_coverage",
                  alpha=0.1, figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="robustness_perf",
                  metric="MSE", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="robustness_perf_worst",
                  alpha=0.3, figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="resilience_perf",
                  resilience_method="worst-sample", immu_feature=None, metric="MAE", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="resilience_distance",
                  resilience_method="worst-sample", metric="MAE", alpha=0.3, figsize=(5, 4))

