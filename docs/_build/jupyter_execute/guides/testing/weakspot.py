#!/usr/bin/env python
# coding: utf-8

# In[ ]:


results = exp.model_diagnose(model="XGB2", show="weakspot", slice_method="histogram",
                        slice_features=["PAY_1"], threshold=1.1, min_samples=100, metric="ACC",
                        use_test=True, return_data=True, figsize=(5, 4))


# In[ ]:


results = exp.model_diagnose(model="XGB2", show="weakspot", slice_method="histogram",
                       slice_features=["PAY_1", "PAY_2"], threshold=1.1, min_samples=100, metric="ACC", use_test=True, return_data=True, figsize=(5, 4))

