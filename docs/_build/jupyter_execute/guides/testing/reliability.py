#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.model_diagnose(model="XGB2", show="reliability_table", alpha=0.1)


# In[ ]:


exp.model_diagnose(model="XGB2", show="reliability_distance", alpha=0.1,
                   threshold=1.1, distance_metric="PSI", figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="XGB2", show="reliability_marginal", alpha=0.1,
                   show_feature="hr", bins=10, threshold=1.1,
                   original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="XGB2", show="reliability_distance",
                   threshold=1.1, distance_metric="PSI", figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="GAM", show="reliability_marginal",
                   show_feature="PAY_1", bins=10, threshold=1.1, figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="GAM", show="reliability_calibration", figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="GAM", show="reliability_perf", figsize=(5, 4))


# In[ ]:


results = exp.model_diagnose(model="GAM", show="reliability_table", return_data=True)
results.data

