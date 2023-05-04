#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.data_loader(data="CoCircles")


# In[ ]:


data_pandas = pd.read_csv('https://github.com/SelfExplainML/PiML-Toolbox/blob/main/datasets/BikeSharing.csv?raw=true')
exp.data_loader(data=data_pandas)

