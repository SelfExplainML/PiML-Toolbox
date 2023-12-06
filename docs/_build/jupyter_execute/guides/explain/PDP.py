#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.model_explain(model="XGB2", show="pdp", uni_feature="hr", use_test=False,
                  grid_size=50, original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_explain(model="XGB2", show="pdp", uni_feature="season",
                  original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_explain(model="XGB2", show="pdp", bi_features=["hr", "workingday"],
                  grid_size=10, sample_size=10000, sliced_line=False, original_scale=True, figsize=(5, 4))

