#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.twosample_test(metric="PSI", psi_buckets="uniform", figsize=(5, 4))


# In[ ]:


exp.twosample_test(metric="PSI", psi_buckets='quantile', feature="atemp", figsize=(5, 4))

