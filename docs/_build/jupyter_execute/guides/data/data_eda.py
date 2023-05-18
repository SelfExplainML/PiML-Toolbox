#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.eda(show='univariate', uni_feature='cnt', figsize=(5, 4))


# In[ ]:


exp.eda(show='univariate', uni_feature='yr', figsize=(5, 4))


# In[ ]:


exp.eda(show='bivariate', bi_features=['temp', 'cnt'], figsize=(5, 4))


# In[ ]:


exp.eda(show='bivariate', bi_features=['hr', 'season'], figsize=(5, 4))


# In[ ]:


exp.eda(show='bivariate', bi_features=['yr', 'season'], figsize=(5, 4))


# In[ ]:


exp.eda(show='multivariate', multi_type='correlation_heatmap', figsize=(6, 5))


# In[ ]:


exp.eda(show='multivariate', multi_type='correlation_graph', figsize=(6, 5))

