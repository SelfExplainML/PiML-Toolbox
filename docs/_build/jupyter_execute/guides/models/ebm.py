#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from piml.models import ExplainableBoostingRegressor
exp.model_train(model=ExplainableBoostingRegressor(interactions=10), name="EBM")


# In[ ]:


exp.model_interpret(model="EBM", show="global_effect_plot", uni_feature="hr",
                    original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="EBM", show="global_effect_plot", bi_features=["hr", "season"],
                    original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="EBM", show="global_ei", figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="EBM", show="global_fi", figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="EBM", show="local_ei", sample_id=0, original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="EBM", show="local_fi", sample_id=0, original_scale=True, figsize=(5, 4))

