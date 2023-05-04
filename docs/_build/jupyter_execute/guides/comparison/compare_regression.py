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
                  slice_method="historgram", slice_feature="hr", threshold=1.05,
                  bins=10, metricmetric="MSE", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="overfit",
                  slice_method="tree", slice_feature="atemp", threshold=1.05,
                  metricmetric="ACC", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="reliability_coverage",
                  alpha=0.1, figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="reliability_bandwidth",
                  alpha=0.1, figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="robustness_perf",
                  perturb_method="raw", perturb_size=0.2, metric="MSE", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="robustness_perf_worst",
                  perturb_method="quantile", perturb_size=0.1, metric="R2", alpha=0.3, figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="resilience_perf",
                  resilience_method="worst-sample", immu_feature=None, metric="MSE", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["GLM", "XGB2", "XGB7"], show="resilience_perf_worst",
                  resilience_method="worst-sample", immu_feature=None, metric="MSE",
                  alpha=0.3, figsize=(5, 4))

