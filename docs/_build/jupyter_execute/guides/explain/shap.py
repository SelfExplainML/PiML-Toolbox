#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.model_explain(model="XGB2", show="shap_waterfall", sample_id=0, figsize=(5, 4))


# In[ ]:


exp.model_explain(model="XGB2", show="shap_fi", sample_size=100, figsize=(5, 4))


# In[ ]:


exp.model_explain(model="XGB2", show="shap_summary", original_scale=True, sample_size=100, figsize=(5, 4))


# In[ ]:


exp.model_explain(model="XGB2", show="shap_scatter", uni_feature="hr",
                  sample_size=100,  original_scale=True, figsize=(5, 4))

