# -*- coding: utf-8 -*-
"""
HPO - Random Search
=========================================
"""
#%%
# Experiment initialization and data preparation
import scipy
from piml import Experiment
from piml.models import GLMClassifier

exp = Experiment()
exp.data_loader("SimuCredit", silent=True)
exp.data_summary(feature_exclude=["Race", "Gender"], silent=True)
exp.data_prepare(target="Approved", task_type="classification", silent=True)

#%%
# Train Model
exp.model_train(model=GLMClassifier(), name="GLM")

#%%
# Define hyperparameter search space for grid search
parameters = {'l1_regularization': scipy.stats.uniform(0, 0.1),
              'l2_regularization': scipy.stats.uniform(0, 0.1)}

#%%
# Tune hyperparameters of registered models
result = exp.model_tune("GLM", method="randomized", parameters=parameters, n_runs=100,
                        metric="AUC", test_ratio=0.2)
result.data

#%%
# Refit model using a selected hyperparameter
params = result.get_params_ranks(rank=1)
exp.model_train(GLMClassifier(**params), name="GLM-HPO-RandSearch")

#%%
# Compare the default model and HPO refitted model
exp.model_diagnose("GLM", show="accuracy_table")

#%%
# Compare the default model and HPO refitted model
exp.model_diagnose("GLM-HPO-RandSearch", show="accuracy_table")
