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
                  slice_method="histogram", slice_feature="PAY_1", threshold=1.05,
                  bins=10, metricmetric="ACC", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="overfit",
                  slice_method="tree", slice_feature="PAY_1", threshold=1.05,
                  metricmetric="ACC", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="reliability_bandwidth", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="reliability_perf",
                  bins=10, figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="robustness_perf",
                  perturb_method="raw", perturb_size=0.2, metric="AUC", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="robustness_perf_worst",
                  perturb_method="quantile", perturb_size=0.1, metric="F1", alpha=0.3, figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="resilience_perf",
                  resilience_method="worst-sample", immu_feature=None, metric="AUC", figsize=(5, 4))


# In[ ]:


exp.model_compare(models=["Tree", "FIGS", "XGB2"], show="resilience_perf_worst",
                  resilience_method="worst-sample", immu_feature=None, metric="AUC",
                  alpha=0.3, figsize=(5, 4))

