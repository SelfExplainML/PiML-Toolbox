# -*- coding: utf-8 -*-
"""
ReLU DNN Regression (Friedman)
=====================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import ReluDNNRegressor

exp = Experiment()
exp.data_loader(data="Friedman", silent=True)
exp.data_summary(silent=True)
exp.data_prepare(target="target", task_type="regression", silent=True)

#%%
# Train Model
exp.model_train(model=ReluDNNRegressor(hidden_layer_sizes=(40, 40), l1_reg=0.0002, learning_rate=0.001),
                name="ReLU-DNN")

#%%
# Evaluate predictive performance
exp.model_diagnose(model='ReLU-DNN', show="accuracy_table")

#%%
#Local Linear Model (LLM) summary plot
exp.model_interpret(model="ReLU-DNN", show="llm_summary", figsize=(5, 4))
#%%
#Local Linear Model (LLM) parallel coordinate plot
exp.model_interpret(model="ReLU-DNN", show="llm_pc", figsize=(5, 4))
#%%
#Local Linear Model (LLM) violin plot
exp.model_interpret(model="ReLU-DNN", show="llm_violin", figsize=(5, 4))
#%%
# Global feature importance 
exp.model_interpret(model="ReLU-DNN", show="global_fi", figsize=(5, 4))
#%%
# Global effect plot with one feature
exp.model_interpret(model="ReLU-DNN", show="global_effect_plot", uni_feature="X0",
                    original_scale=True, figsize=(5, 4))

#%%
# Global effect plot with two features
exp.model_interpret(model="ReLU-DNN", show="global_effect_plot", bi_features=["X0", "X2"],
                    original_scale=True, figsize=(5, 4))

#%%
# Local feature importance without centering
exp.model_interpret(model="ReLU-DNN", show="local_fi", sample_id=0, centered=False,
                    original_scale=True, figsize=(5, 4))
#%%
# Local feature importance with centering
exp.model_interpret(model="ReLU-DNN", show="local_fi", sample_id=0, centered=True, 
                    original_scale=True, figsize=(5, 4))
