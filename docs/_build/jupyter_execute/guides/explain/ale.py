#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.model_explain(model="ReLUDNN", show="ale", uni_feature="hr",
                  original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_explain(model="ReLUDNN", show="ale", uni_feature="atemp",
                  original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_explain(model="ReLUDNN", show="ale", uni_feature="weathersit",
                  original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_explain(model="ReLUDNN", show="ale", bi_features=["hr", "atemp"]
                  original_scale=True, figsize=(5, 4))

