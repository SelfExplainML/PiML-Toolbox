#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from piml.models import XGB1Regressor
exp.model_train(model=XGB1Regressor(n_estimators=100, max_bin=20, min_bin_size=0.01), name="XGB1")


# In[ ]:


exp.model_interpret(model="XGB1", show="global_effect_plot", uni_feature="MedInc",
                    original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="XGB1", show="global_fi", figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="XGB1", show="xgb1_woe", uni_feature="MedInc",
                   original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="XGB1", show="xgb1_iv", figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="XGB1", show="local_fi", sample_id=0, original_scale=True, figsize=(5, 4))

