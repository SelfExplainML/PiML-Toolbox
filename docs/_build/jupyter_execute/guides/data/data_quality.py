#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from piml.data.outlier_detection import PCA
exp.data_quality_check(method=PCA(), show='score_distribution', threshold=0.999)


# In[ ]:


from piml.data.outlier_detection import PCA
exp.data_quality_check(method=PCA(), show='marginal_outlier_distribution', threshold=0.999)


# In[ ]:


from piml.data.outlier_detection import PCA, CBLOF
exp.data_quality_check(method=[PCA(), CBLOF()], show='tsne_comparison', threshold=[0.999, 0.999])

