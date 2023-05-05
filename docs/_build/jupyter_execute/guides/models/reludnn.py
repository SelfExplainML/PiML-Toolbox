#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from piml.models import ReluDNNClassifier
exp.model_train(model=ReluDNNClassifier(hidden_layer_sizes=(40, 40), l1_reg=0.0002, learning_rate=0.001),
                name="ReLUDNN")


# In[ ]:


exp.model_interpret(model="ReLUDNN", show="llm_summary")


# In[ ]:


exp.model_interpret(model="ReLUDNN", show="llm_pc", figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="ReLUDNN", show="llm_violin", figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="ReLUDNN", show="global_fi", figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="ReLUDNN", show="global_effect_plot", uni_feature="PAY_1", original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="ReLUDNN", show="global_effect_plot", bi_features=["PAY_1", "PAY_3"], original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="ReLUDNN", show="local_fi", sample_id=0, centered=False, original_scale=True, figsize=(5, 4))


# In[ ]:


exp.model_interpret(model="ReLUDNN", show="local_fi", sample_id=0, centered=True, original_scale=True, figsize=(5, 4))

