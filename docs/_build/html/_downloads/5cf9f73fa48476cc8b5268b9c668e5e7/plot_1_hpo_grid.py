# -*- coding: utf-8 -*-
"""
HPO - XGB - Grid Search (Bike Sharing)
=========================================
"""
#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import XGB2Regressor

exp = Experiment()
exp.data_loader(data="BikeSharing", silent=True)
exp.data_summary(feature_exclude=["yr", "mnth", "temp"], silent=True)
exp.data_prepare(target="cnt", task_type="regression", silent=True)

#%%
# Train Model
exp.model_train(model=XGB2Regressor(), name="XGB2")

#%%
# Define hyperparameter search space for grid search
parameters = {'n_estimators': [100, 300, 500],
              'eta': [0.1, 0.3, 0.5],
              'reg_lambda': [0.0, 0.5, 1.0],
              'reg_alpha': [0.0, 0.5, 1.0]
             }

#%%
# Tune hyperparameters of registered models
result = exp.model_tune("XGB2", method="grid", parameters=parameters, metric=['MSE', 'MAE'], test_ratio=0.2)
result.data

#%%
# Show hyperparameter result plot
fig = result.plot(param='n_estimators', figsize=(6, 4.5))

#%%
# Refit model using a selected hyperparameter
params = result.get_params_ranks(rank=1)
exp.model_train(XGB2Regressor(**params), name="XGB2-HPO-GridSearch")

#%%
# Compare the default model and HPO refitted model
exp.model_diagnose("XGB2", show="accuracy_table")

#%%
# Compare the default model and HPO refitted model
exp.model_diagnose("XGB2-HPO-GridSearch", show="accuracy_table")