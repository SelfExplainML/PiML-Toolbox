#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.data_quality(show="drift_test_distance", distance_metric="PSI",
                 psi_buckets="uniform", figsize=(5, 4))


# In[ ]:


exp.data_quality(show="drift_test_distance", distance_metric="PSI", psi_buckets='quantile',
                 show_feature="atemp", figsize=(5, 4))


# In[ ]:


exp.data_quality(show="drift_test_info", figsize=(5, 4))

