#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from piml.models import GAMINetRegressor
exp.model_train(model=GAMINetRegressor(), name="GAMI-Net")


# In[ ]:


exp.model_interpret(model="GAMI-Net", show="global_effect_plot", uni_feature="hr",
                   original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="GAMI-Net", show="global_effect_plot", bi_features=["hr", "weekday"],
                    original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="GAMI-Net", show="global_ei", figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="GAMI-Net", show="global_fi", figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="GAMI-Net", show="local_ei", sample_id=0, original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="GAMI-Net", show="local_fi", sample_id=0, original_scale=True, figsize=(5, 4))

