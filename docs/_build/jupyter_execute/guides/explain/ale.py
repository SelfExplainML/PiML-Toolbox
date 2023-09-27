#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.model_explain(model="ReLUDNN", show="ale", uni_feature="hr",
                  grid_size=50, original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_explain(model="ReLUDNN", show="ale", uni_feature="atemp",
                  grid_size=50, original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_explain(model="ReLUDNN", show="ale", uni_feature="weathersit",
                  original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_explain(model="ReLUDNN", show="ale", bi_features=["hr", "atemp"]
                  grid_size=10, sliced_line=False, original_scale=True, figsize=(5, 4))

