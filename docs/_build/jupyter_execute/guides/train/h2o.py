#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import h2o
h2o.no_progress()
h2o.init(verbose=False)

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from h2o.estimators import H2OGradientBoostingEstimator

data = fetch_california_housing()
feature_names = data.feature_names
target_name = data.target_names[0]

h2o_data = h2o.H2OFrame(pd.DataFrame(np.hstack([data.data, data.target.reshape(-1, 1)]),
                                    columns=feature_names + [target_name]))
h2o_data_train, h2o_data_test = h2o_data.split_frame(ratios=[0.8], seed=2023)

gbm_model = H2OGradientBoostingEstimator()
gbm_model.train(feature_names, target_name, training_frame=h2o_data_train)


# In[ ]:


mojo_file_path = gbm_model.save_mojo(path="./")


# In[ ]:


from piml import Experiment
exp = Experiment(highcode_only=True)

imported_model = h2o.import_mojo(mojo_file_path)
pipeline = exp.make_pipeline(model=imported_model,
                             task_type="regression",
                             train_x=h2o_data_train[feature_names].as_data_frame().values,
                             train_y=h2o_data_train[target_name].as_data_frame().values.ravel(),
                             test_x=h2o_data_test[feature_names].as_data_frame().values,
                             test_y=h2o_data_test[target_name].as_data_frame().values.ravel(),
                             feature_names=feature_names,
                             target_name=target_name)
exp.register(pipeline, "H2O-GBM")


# In[ ]:


exp.model_explain(model="H2O-GBM", show="pfi", figsize=(5, 4))

