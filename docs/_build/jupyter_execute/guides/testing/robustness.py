#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.model_diagnose(model="FIGS", show='robustness_perf', perturb_features=None,
                  perturb_method="raw", metric="MSE", perturb_size=0.1, figsize=(6, 4))


# In[ ]:


exp.model_diagnose(model="FIGS", show="robustness_perf", perturb_features=["hr", "atemp"],
                  perturb_method='raw', metric="MSE", perturb_size=0.1,  figsize=(6, 4))


# In[ ]:


exp.model_diagnose(model="FIGS", show="robustness_perf_worst", perturb_features=None,
                  perturb_method="raw", metric="MSE", perturb_size=0.1, alpha=0.3, figsize=(6, 4))

