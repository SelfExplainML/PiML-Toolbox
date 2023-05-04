#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.model_explain(model="XGB2", show="pdp", uni_feature="hr",
                  original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_explain(model="XGB2", show="pdp", uni_feature="season",
                 original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_explain(model="XGB2", show="pdp", bi_features=["hr", "workingday"],
                  pdp_size=10000, original_scale=True, figsize=(5, 4))

