#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.feature_select(method="cor", corr_algorithm="pearson", threshold=0.1, figsize=(5, 4))


# In[ ]:


exp.feature_select(method="cor", corr_algorithm="spearman", threshold=0.1, figsize=(5, 4))


# In[ ]:


exp.feature_select(method="dcor", threshold=0.1, figsize=(5, 4))


# In[ ]:


exp.feature_select(method="pfi", threshold=0.95, figsize=(5, 4))


# In[ ]:


exp.feature_select(method="rcit", threshold=0.001, n_forward_phase=2, kernel_size=100, figsize=(5, 4))


# In[ ]:


exp.feature_select(method="rcit", threshold=0.001, preset=["hr", "temp"], figsize=(5, 4))

