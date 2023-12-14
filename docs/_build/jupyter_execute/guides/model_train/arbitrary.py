#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import statsmodels.api as sm

x = np.random.uniform(-1, 1, size=(1000, 2))
y = (np.sum(x, axis=1) + np.random.normal(0, 0.1, size=(1000,))) > 0.0

glm_binom = sm.GLM(y, x, family=sm.families.Binomial())
glm_results  = glm_binom.fit()


# In[ ]:


def predict_proba_func(X):
    proba = glm_binom.predict(glm_results.params, exog=X)
    return np.vstack([1 - proba, proba]).T

def predict_func(X):
    proba = glm_binom.predict(glm_results.params, exog=X)
    return proba > 0.5


# In[ ]:


from piml import Experiment
exp = Experiment(highcode_only=True)

pipeline = exp.make_pipeline(predict_func=predict_func,
                             predict_proba_func=predict_proba_func,
                             task_type="classification",
                             train_x=x[:800],
                             train_y=y[:800],
                             test_x=x[800:],
                             test_y=y[800:],
                             feature_names=["X0", "X1"],
                             target_name="Y")
exp.register(pipeline, "Statsmodels-GLM")


# In[ ]:


exp.model_explain(model="Statsmodels-GLM", show="ale", uni_feature="X0", figsize=(5, 4))

