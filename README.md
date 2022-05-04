<div align="center">
  
<img src="https://github.com/SelfExplainML/PiML-Toolbox/blob/main/examples/results/LogoPiML.png" alt="drawing" width="314.15926"/>

**A low-code interpretable machine learning toolbox in Python** 
</div>

PiML (or π·ML, /ˈpaɪ ˈem ˈel/) is a new Python toolbox for Interpretable Machine Learning model development and validation. Through low-code automation and high-code programming, PiML supports various machine learning models in the following two categories:

- **Inherently interpretable models**: 
  1. EBM: Explainable Boosting Machine (Nori, et al. 2019; Lou, et al. 2013)
  2. GAMI-Net: Generalized Additive Model with Structured Interactions (Yang, Zhang and Sudjianto, 2021)
  3. ReLU-DNN: Deep ReLU Networks using Aletheia Unwrapper (Sudjianto, et al. 2020)

- **Arbitrary black-box models**，e.g.
  1. LightGBM or XGBoost of varying depth
  2. RandomForest of varying depth
  3. Residual Deep Neural Networks

## Installation 

Run the following piece of sript to download and install PiML v0.1.0 to Google Colab: 

```python
!pip install wget
import wget
url = "https://github.com/SelfExplainML/PiML-Toolbox/releases/download/V0.1.0/PiML-0.1.0-cp37-cp37m-linux_x86_64.whl"
wget.download(url, 'PiML-0.1.0-cp37-cp37m-linux_x86_64.whl')
!pip install PiML-0.1.0-cp37-cp37m-linux_x86_64.whl
```

## Low-code Usage on Google Colab

### Stage 1:  Initialize an experiment, Load and Prepare data

```python
from piml import Experiment
exp = Experiment(platform="colab")
```

```python
exp.data_loader()
```
<img src="https://github.com/SelfExplainML/PiML-Toolbox/blob/main/examples/results/data_loader.png">

```python
exp.data_summary()
```
<img src="https://github.com/SelfExplainML/PiML-Toolbox/blob/main/examples/results/data_summary.png">

```python
exp.data_prepare()
```
<img src="https://github.com/SelfExplainML/PiML-Toolbox/blob/main/examples/results/data_prepare.png">

```python
exp.eda()
```
<img src="https://github.com/SelfExplainML/PiML-Toolbox/blob/main/examples/results/data_eda.png">

### Stage 2:  Train intepretable models
```python
exp.model_train()
```
<img src="https://github.com/SelfExplainML/PiML-Toolbox/blob/main/examples/results/model_train.png">


### Stage 3. Explain and Interpret
```python
exp.model_explain()
```
<img src="https://github.com/SelfExplainML/PiML-Toolbox/blob/main/examples/results/model_explain.png">

```python
exp.model_interpret()
```
<img src="https://github.com/SelfExplainML/PiML-Toolbox/blob/main/examples/results/model_interpret.png">

### Stage 4. Diagnose and Compare
```python
exp.model_diagnose()
```
<img src="https://github.com/SelfExplainML/PiML-Toolbox/blob/main/examples/results/model_diagnose.png">

```python
exp.model_compare()
```
<img src="https://github.com/SelfExplainML/PiML-Toolbox/blob/main/examples/results/model_compare.png">



## Arbitrary Black-Box Modeling
For example, train a complex LightGBM with depth 7 and register it to the experiment: 

```python
from lightgbm import LGBMRegressor
pipeline = exp.make_pipeline(LGBMRegressor(max_depth=7))
pipeline.fit() 
exp.register(pipeline=pipeline, name='LGBM')
```

Then, compare it to inherently interpretable models (e.g. EBM and GAMI-Net): 
```python
exp.model_compare()
```
<img src="https://github.com/SelfExplainML/PiML-Toolbox/blob/main/examples/results/model_compare2.png">



## Citations
To be added ... 


