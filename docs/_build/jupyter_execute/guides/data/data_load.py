#!/usr/bin/env python
# coding: utf-8

# In[ ]:


exp.data_loader(data="CoCircles")


# In[ ]:


data = pd.read_csv('https://github.com/SelfExplainML/PiML-Toolbox/blob/main/datasets/BikeSharing.csv?raw=true')
exp.data_loader(data=data)


# In[ ]:


exp.data_loader(data="./myfile.parquet", spark=True, spark_sample_size=10000, spark_random_state=0)


# In[ ]:


exp.data_loader(data="./myfile.parquet", spark=True, spark_sample_size=10000,
                spark_sample_by_feature='Y', spark_random_state=0)


# In[ ]:


exp.data_loader(data="./myfile.parquet", spark=True, spark_sample_size=10000,
                spark_sample_by_feature='Y', spark_sample_fractions={0.0: 1, 1.0: 5},
                spark_random_state=0)

