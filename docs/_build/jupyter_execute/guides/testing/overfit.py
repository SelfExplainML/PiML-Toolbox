#!/usr/bin/env python
# coding: utf-8

# In[ ]:


results = exp.model_diagnose(model="XGB2", show="overfit", slice_method="histogram",
                             slice_features=["hr"], threshold=1.05, min_samples=100,
                             return_data=True, figsize=(5, 4))


# In[ ]:


results=exp.model_diagnose(model="XGB2", show="overfit", slice_method="histogram",
                           slice_features=["hr", "atemp"], threshold=1.05, min_samples=100,
                           return_data=True, figsize=(5, 4))

