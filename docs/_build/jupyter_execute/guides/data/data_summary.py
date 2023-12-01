#!/usr/bin/env python
# coding: utf-8

# In[1]:


from piml import Experiment

exp = Experiment()
exp.data_loader(data="BikeSharing", silent=True)


# In[2]:


exp.data_summary(feature_exclude=[], feature_type={})


# In[3]:


exp.data_summary(feature_exclude=["yr", "mnth", "temp"])


# In[4]:


exp.data_summary(feature_exclude=["yr", "mnth", "temp"], feature_type={"weekday": "categorical"})

