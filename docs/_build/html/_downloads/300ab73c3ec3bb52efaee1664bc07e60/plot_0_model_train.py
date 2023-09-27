# -*- coding: utf-8 -*-
"""
Train External Models 
========================================================
"""

#%%
# Load PiML
from piml import Experiment
exp = Experiment()
exp.data_loader(data='CaliforniaHousing_raw')
exp.data_prepare(target='MedHouseVal', task_type='regression', random_state=0)

#%%
# Train and Register Models using piml
from lightgbm import LGBMRegressor
lgbm2 = LGBMRegressor(max_depth=2)
exp.model_train(lgbm2, name='LGBM_2')


#%%
# Save Fitted Models
exp.model_save("LGBM_2", "CH_LGBM_2.pkl")


#%%
# Load model from file system, if not specified, the default train and test data will be used. 
pipeline = exp.make_pipeline(model='CH_LGBM_2.pkl')
exp.register(pipeline, "LGBM_2_load")

#%%
# Run post-hoc explanation using PDP.
exp.model_explain(model="LGBM_2_load", show="pdp", uni_feature="MedInc", figsize=(5, 4))
