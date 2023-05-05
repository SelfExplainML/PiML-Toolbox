#!/usr/bin/env python
# coding: utf-8

# In[ ]:


custom_train_idx = np.arange(0,16000)
custom_test_idx = np.arange(16000, 17379)
exp.data_prepare(train_idx=custom_train_idx, test_idx=custom_test_idx)


# In[ ]:


exp.data_prepare(target='cnt', task_type='Regression', sample_weight=None,
                 split_method='random', test_ratio=0.2, random_state=0)

