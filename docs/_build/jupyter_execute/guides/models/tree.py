#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from piml.models import TreeRegressor
exp.model_train(model=TreeRegressor(max_depth=6), name="Tree")


# In[ ]:


exp.model_interpret(model="Tree", show="tree_global", root=0,
                    depth=3, original_scale=True, figsize=(16, 10))


# In[ ]:


exp.model_interpret(model="Tree", show="tree_local", sample_id=0, original_scale=True, figsize=(16, 10))

