#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from piml import Experiment
exp = Experiment()
exp.data_loader(data='CaliforniaHousing_raw')
exp.data_prepare(target='MedHouseVal', task_type='regression', random_state=0)


# In[ ]:


from lightgbm import LGBMRegressor
lgbm2 = LGBMRegressor(max_depth=2)
exp.model_train(lgbm2, name='LGBM_2')


# In[ ]:


exp.model_save("LGBM_2", "CH_LGBM_2.pkl")


# In[ ]:


pipeline = exp.make_pipeline(model='CH_LGBM_2.pkl')
exp.register(pipeline, "LGBM_2_load")


# In[ ]:


from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(exp.dataset.x, exp.dataset.y, test_size=0.2)

lgbm7 = LGBMRegressor(max_depth=7, n_estimators=100)
lgbm7.fit(train_x, train_y)

pipeline = exp.make_pipeline(model=lgbm7, train_x=train_x, train_y=train_y.ravel(),
                             test_x=test_x, test_y=test_y.ravel())
exp.register(pipeline, "LGBM_7")


# In[ ]:


exp.model_explain(model="LGBM_7", show="pdp", uni_feature="MedInc", figsize=(5, 4))

