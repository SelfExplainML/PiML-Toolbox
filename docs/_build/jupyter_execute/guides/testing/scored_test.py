#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from piml.scored_test import test_accuracy_residual
result = test_accuracy_residual(**data_dict, show_feature='MedInc', figsize=(5, 4))

