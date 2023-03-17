#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from piml import Experiment
from piml.models import ReluDNNRegressor
exp = Experiment()
exp.data_loader(data='BikeSharing')
exp.data_summary(feature_exclude=["season", "workingday", "atemp"])
exp.data_prepare(target='cnt', task_type='Regression', test_ratio=0.2, random_state=0)
exp.model_train(model=ReluDNNRegressor(hidden_layer_sizes=(40, 40), l1_reg=1e-05,
                                     batch_size=500, learning_rate=0.001), name='ReLU_DNN')


# In[ ]:


exp.model_explain(model='ReLU_DNN', show='pdp', original_scale=True, uni_feature='hr', figsize=(6, 5))


# In[ ]:


exp.model_explain(model='ReLU_DNN', show='pdp', original_scale=True, uni_feature='holiday', figsize=(6, 5))


# In[ ]:


exp.model_explain(model='ReLU_DNN', show='pdp', original_scale=True,
                                      bi_features=['hr', 'weathersit'], figsize=(6, 5))

