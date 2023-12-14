#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="accuracy_plot", metric="ACC")


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="accuracy_plot", metric="AUC")


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="accuracy_plot", metric="F1")


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="overfit",
                  slice_method="histogram", slice_feature="PAY_1",
                  bins=10, metric="ACC", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="overfit",
                  slice_method="histogram", slice_feature="PAY_1",
                  metric="AUC", original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="reliability_bandwidth", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="reliability_perf",
                  bins=10, figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="robustness_perf",
                  figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="robustness_perf_worst",
                  alpha=0.3, figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="resilience_perf",
                  resilience_method="worst-sample", immu_feature=None, metric="AUC", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="resilience_distance",
                  resilience_method="worst-sample", metric="AUC", alpha=0.3, figsize=(5, 4))

