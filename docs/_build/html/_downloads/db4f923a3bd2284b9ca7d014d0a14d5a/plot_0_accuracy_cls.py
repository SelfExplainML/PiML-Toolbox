# -*- coding: utf-8 -*-
"""
Accuracy: Classification
=====================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import XGB2Classifier

exp = Experiment()
exp.data_loader(data="TaiwanCredit", silent=True)
exp.data_summary(feature_exclude=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"], silent=True)
exp.data_prepare(target="FlagDefault", task_type="classification", silent=True)

#%%
# Train Model
exp.model_train(XGB2Classifier(), name="XGB2")
#%%
#Accuracy table
exp.model_diagnose(model="XGB2", show="accuracy_table")
#%%
#Plot confusion matrix, ROC and Recall-Precision   
exp.model_diagnose(model="XGB2", show="accuracy_plot", figsize=(10, 4))

#%%
#Plot residual with respect to the feature PAY_1
exp.model_diagnose(model="XGB2", show="accuracy_residual", show_feature="PAY_1",
                   use_test=False, original_scale=True, figsize=(5, 4))
#%%
#Plot residual with respect to the target feature
exp.model_diagnose(model="XGB2", show="accuracy_residual", show_feature="FlagDefault",
                   use_test=False, figsize=(5, 4))
#%%
#Plot residual with respect to the predicted response
exp.model_diagnose(model="XGB2", show="accuracy_residual", show_feature="FlagDefault_predict",
                   use_test=False, figsize=(5, 4))
