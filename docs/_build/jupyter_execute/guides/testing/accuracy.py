#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.model_diagnose(model="XGB2", show='accuracy_table')


# In[ ]:


exp.model_diagnose(model="XGB2", show="accuracy_residual", show_feature="hr",
                   use_test=False, original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="XGB2", show="accuracy_residual", show_feature="season",
                   use_test=False, original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="XGB2", show="accuracy_residual", show_feature="cnt",
                   use_test=False, figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="XGB2", show="accuracy_residual", show_feature="cnt_predict",
                   use_test=False, figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="XGB2", show="accuracy_table")


# In[ ]:


exp.model_diagnose(model="XGB2", show="residual_plot", show_feature="PAY_1",
                   use_test=False, original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_diagnose(model="XGB2", show="accuracy_plot")

