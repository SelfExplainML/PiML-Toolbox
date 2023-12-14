#!/usr/bin/env python
# coding: utf-8

# In[ ]:


parameters = {'n_estimators': [100, 300, 500],
              'eta': [0.1, 0.3, 0.5],
              'reg_lambda': [0.0, 0.5, 1.0],
              'reg_alpha': [0.0, 0.5, 1.0]}
result = exp.model_tune("XGB2", method="grid", parameters=parameters, metric=['MSE', 'MAE'], test_ratio=0.2)


# In[ ]:


result.data


# In[ ]:


result.plot(param='n_estimators', figsize=(6, 4.5))


# In[ ]:


params = result.get_params_ranks(rank=1)
exp.model_train(XGB2Regressor(**params), name="XGB2-HPO-GridSearch")

