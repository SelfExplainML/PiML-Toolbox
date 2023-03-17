# -*- coding: utf-8 -*-
"""
SHAP:  Classification
=====================================

SHAP plots on classification problem
"""

#%%
from piml import Experiment
from piml.models import ReluDNNClassifier
exp = Experiment()
exp.data_loader(data='TaiwanCredit')
exp.data_summary(feature_exclude=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"], feature_type={})
exp.data_prepare(target='FlagDefault', task_type='Classification', test_ratio=0.2, random_state=0)
classifier = ReluDNNClassifier(hidden_layer_sizes=(20, 20), l1_reg=0.0008, batch_size=500, learning_rate=0.001)
exp.model_train(model=classifier, name='ReLU-DNN')
#%%
# Plot SHAP 
exp.model_explain(model='ReLU-DNN', show='shap_waterfall')
#%%
exp.model_explain(model='ReLU-DNN', show='shap_fi', sample_size=10)
#%%
exp.model_explain(model='ReLU-DNN', show='shap_summary', sample_size=10)
#%%
exp.model_explain(model='ReLU-DNN', uni_feature='BILL_AMT1', show='shap_scatter', sample_size=10)
