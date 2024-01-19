# -*- coding: utf-8 -*-
"""
Build Robust Models with Monotonicity Constraints
========================================================
"""

#%%
# Experiment initialization and data preparation
from piml import Experiment
from piml.models import XGB2Classifier, GAMINetClassifier

exp = Experiment()
exp.data_loader(data="SimuCredit", silent=True)
exp.data_summary(feature_exclude=["Gender", "Race"], silent=True)
exp.data_prepare(target="Approved", task_type="classification", silent=True)

#%%
# Train XGB2 without monotonicity constraints 
exp.model_train(XGB2Classifier(n_estimators=500), name="XGB2")

#%%
# Train XGB2 with monotonicity constraints on Balance
exp.model_train(model=XGB2Classifier(n_estimators=500, mono_increasing_list=("Balance", )), name="Mono-XGB2")

#%%
# Train GAMI-Net with monotonicity constraints on Balance
exp.model_train(model=GAMINetClassifier(mono_increasing_list=("Balance", )), name="Mono-GAMI-Net")

#%%
# Main effect plot of XGB2 on Balance
exp.model_interpret(model="XGB2", show="global_effect_plot", uni_feature="Balance", figsize=(5, 4))

#%%
# Main effect plot of Mono-XGB2 on Balance
exp.model_interpret(model="Mono-XGB2", show="global_effect_plot", uni_feature="Balance", figsize=(5, 4))

#%%
# Main effect plot of Mono-GAMI-Net on Balance
exp.model_interpret(model="Mono-GAMI-Net", show="global_effect_plot", uni_feature="Balance", figsize=(5, 4))

#%%
# Robustness comparison with default settings
exp.model_compare(models=["XGB2", "Mono-XGB2", "Mono-GAMI-Net"], show="robustness_perf", figsize=(6, 4))
