# -*- coding: utf-8 -*-
"""
ReLU DNN Classification (Taiwan Credit)
=========================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import ReluDNNClassifier

exp = Experiment()
exp.data_loader(data="TaiwanCredit", silent=True)
exp.data_summary(feature_exclude=["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"], silent=True)
exp.data_prepare(target="FlagDefault", task_type="classification", silent=True)

#%%
# Train Model
exp.model_train(model=ReluDNNClassifier(hidden_layer_sizes=(40, 40), l1_reg=0.0002, learning_rate=0.001),
                name="ReLUDNN")
#%%
# Evaluate predictive performance
exp.model_diagnose(model='ReLUDNN', show="accuracy_table")

#%%
# Local Linear Model (LLM) summary plot
exp.model_interpret(model="ReLUDNN", show="llm_summary", figsize=(5, 4))
#%%
# Local Linear Model (LLM) parallel coordinate plot
exp.model_interpret(model="ReLUDNN", show="llm_pc", figsize=(5, 4))
#%%
# Local Linear Model (LLM) violin plot
exp.model_interpret(model="ReLUDNN", show="llm_violin", figsize=(5, 4))
#%%
# Global feature importance
exp.model_interpret(model="ReLUDNN", show="global_fi", figsize=(5, 4))
#%%
# Global effect plot: with one feature
exp.model_interpret(model="ReLUDNN", show="global_effect_plot", uni_feature="PAY_1",
                    original_scale=True, figsize=(5, 4))
#%%
# Global effect plot: with two features
exp.model_interpret(model="ReLUDNN", show="global_effect_plot", bi_features=["PAY_1", "PAY_3"],
                    original_scale=True, figsize=(5, 4))

#%%
# Local feature importance without centering
exp.model_interpret(model="ReLUDNN", show="local_fi", sample_id=0, centered=False,
                    original_scale=True, figsize=(5, 4))
#%%
# Local feature importance with centering
exp.model_interpret(model="ReLUDNN", show="local_fi", sample_id=0, centered=True,
                    original_scale=True, figsize=(5, 4))
