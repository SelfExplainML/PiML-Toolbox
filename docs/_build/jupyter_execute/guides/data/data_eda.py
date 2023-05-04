#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.eda(show='univariate', uni_feature='cnt')


# In[ ]:


exp.eda(show='univariate', uni_feature='yr')


# In[ ]:


exp.eda(show='bivariate', bi_features=['temp', 'cnt'])


# In[ ]:


exp.eda(show='bivariate', bi_features=['hr', 'season'])


# In[ ]:


exp.eda(show='bivariate', bi_features=['yr', 'season'])


# In[ ]:


exp.eda(show='multivariate', multi_type='correlation_heatmap')


# In[ ]:


exp.eda(show='multivariate', multi_type='correlation_graph')

