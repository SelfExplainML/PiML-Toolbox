#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from piml.models import FIGSRegressor
exp.model_train(model=FIGSRegressor(max_iter=100, max_depth=4), name="FIGS")


# In[ ]:


exp.model_interpret(model="FIGS", show="figs_heatmap", tree_idx=0, figsize=(12, 4))


# In[ ]:


exp.model_interpret(model="FIGS", show="tree_global", tree_idx=0, root=0,
                    depth=3, original_scale=True, figsize=(16, 10))


# In[ ]:


exp.model_interpret(model="FIGS", show="tree_global", tree_idx=1, root=0,
                    depth=3, original_scale=True, figsize=(16, 10))


# In[ ]:


exp.model_interpret(model="FIGS", show="tree_local", sample_id=0, tree_idx=0,
                    original_scale=True, figsize=(16, 10))


# In[ ]:


exp.model_interpret(model="FIGS", show="tree_local", sample_id=0, tree_idx=1,
                    original_scale=True, figsize=(16, 10))

