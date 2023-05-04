#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.model_explain(model="XGB2", show="lime", sample_id=0, centered=False,
                  original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_explain(model="XGB2", show="lime", sample_id=0, centered=True,
                  original_scale=True, figsize=(6,5))

