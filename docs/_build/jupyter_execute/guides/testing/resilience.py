#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.model_diagnose(model="XGB2", show="resilience_perf", resilience_method="worst-sample",
                   metric="AUC", figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="XGB2", show="resilience_perf", resilience_method="hard-sample",
                   metric="AUC", figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="XGB2", show="resilience_perf", resilience_method="outer-sample",
                   metric="AUC", figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="XGB2", show="resilience_perf", resilience_method="worst-cluster",
                   metric="AUC", figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="XGB2", show="resilience_distance", resilience_method="worst-sample",
                   distance_metric="PSI", alpha=0.3, figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="XGB2", show="resilience_distance", resilience_method="worst-sample",
                   distance_metric="PSI", immu_feature="PAY_1", alpha=0.3, figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="XGB2", show="resilience_distance", resilience_method="worst-cluster",
                   distance_metric="WD1", n_clusters=10, figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="XGB2", show="resilience_shift_density",
                   show_feature="Pay_1", original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="XGB2", show="resilience_shift_histogram",
                   show_feature="Pay_1", original_scale=True, figsize=(5, 4))

