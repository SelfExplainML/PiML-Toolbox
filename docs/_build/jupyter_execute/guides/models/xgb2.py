#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from piml.models import XGB2Regressor
exp.model_train(model=XGB2Regressor(), name="XGB2")


# In[ ]:


exp.model_interpret(model="XGB2", show="global_effect_plot", uni_feature="atemp",
                    original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="XGB2", show="global_effect_plot", bi_features=["hr", "workingday"],
                    sliced_line=False, original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="XGB2", show="global_ei", figsize=(5, 4))


# In[ ]:


exp.model_interpret(model='XGB2', show="global_fi", figsize=(5, 4))


# In[ ]:


exp.model_interpret(model='XGB2', show="local_ei", sample_id=0, original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="XGB2", show="local_fi", sample_id=0, original_scale=True, figsize=(5, 4))

