#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from piml.models import GAMRegressor
exp.model_train(model=GAMRegressor(spline_order=1, n_splines=20, lam=0.6), name="GAM")


# In[ ]:


exp.model_interpret(model="GAM", show="global_effect_plot", uni_feature="MedInc",
                    original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="GAM",show="global_fi", figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="GAM",show="local_fi", sample_id=0, original_scale=True, figsize=(5, 4))

