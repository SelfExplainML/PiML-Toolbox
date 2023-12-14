#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from piml.models import GLMRegressor
exp.model_train(model=GLMRegressor(l1_regularization=0.0008, l2_regularization=0.0008), name="GLM")


# In[ ]:


exp.model_diagnose(model="GLM", show="accuracy_table")


# In[ ]:


exp.model_interpret(model="GLM", show="glm_coef_plot", figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="GLM", show="glm_coef_plot", uni_feature="season", figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="GLM", show="glm_coef_table")


# In[ ]:


exp.model_interpret(model="GLM", show="global_fi", figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="GLM", show="local_fi", sample_id=0, centered=False, original_scale=False, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="GLM", show="local_fi", sample_id=0, centered=False, original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="GLM", show="local_fi", sample_id=0, centered=True, original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="GLM", show="global_fi", use_test=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="GLM", show="local_fi", sample_id=0, use_test=True,
                    original_scale=True, figsize=(5, 4))

