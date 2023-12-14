#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.model_explain(model="XGB2", show="hstats", sample_size=2000, grid_size=5,
                  figsize=(5, 4))


# In[ ]:


result = exp.model_explain(model="XGB2", show="hstats", sample_size=2000, grid_size=5,
                           return_data=True, figsize=(5, 4))
result.data

