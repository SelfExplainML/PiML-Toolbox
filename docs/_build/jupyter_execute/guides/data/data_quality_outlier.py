#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from piml.data.outlier_detection import PCA
exp.data_quality(method=PCA(), show='od_score_distribution', threshold=0.999)


# In[ ]:


from piml.data.outlier_detection import PCA
exp.data_quality(method=PCA(), show='od_marginal_outlier_distribution', threshold=0.999)


# In[ ]:


from piml.data.outlier_detection import PCA, CBLOF
exp.data_quality(method=[PCA(), CBLOF()], show='od_tsne_comparison', threshold=[0.999, 0.999])

