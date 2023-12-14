#!/usr/bin/env python
# coding: utf-8

# In[ ]:


result = exp.segmented_diagnose(model='XGB2', show='segment_table',
                                segment_method='uniform', segment_bins=5, return_data=True)
result.data


# In[ ]:


result = exp.segmented_diagnose(model='XGB2', show='accuracy_table', segment_id=0, return_data=True)
result.data.head(10)


# In[ ]:


exp.segmented_diagnose(model='XGB2', show='accuracy_residual', segment_id=0, show_feature='atemp')


# In[ ]:


exp.segmented_diagnose(model='XGB2', show='weakspot', segment_id=0, slice_features=['atemp'])


# In[ ]:


exp.segmented_diagnose(model='XGB2', show='distribution_shift', segment_id=0)


# In[ ]:


exp.segmented_diagnose(model='XGB2', show='distribution_shift', segment_id=0, show_feature='hum')

