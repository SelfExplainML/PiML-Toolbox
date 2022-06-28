## Workshop Info and Materials

### Machine Learning Model Validation for Critical or Regulated Applications

June 29 - July 6, 2022: QU-ML Model Validation Workshop

URL: [https://github.com/SelfExplainML/PiML-Toolbox](https://github.com/SelfExplainML/PiML-Toolbox)

Machine Learning (ML) has gained significant adoption in the industry; however, many concerns remained for their usage in highly regulated or critical applications. As with any models, understanding and testing the risk of ML are front and center in the discipline of model risk management. Given the data driven approach as well as the complexity of ML algorithms, we need to improve the sophistication of model design and validation to evaluate both their conceptual soundness as well as outcome. Key element for the conceptual soundness evaluation is model explainability/interpretability. Comprehensive Machine Learning model validation for real production require analysis beyond the standard model performance evaluation which must cover: identification of model weakness through residual slicing, identification of overfitting and underfitting regions, model robustness under noisy or corrupted inputs, prediction reliability such as evaluation of prediction uncertainty and resilience of model performance under input distribution drift. PiML (Python Interpretable Machine Learning) was created to address all the aforementioned needs in a single easy to use (low code) packages.

We are going to cover the above aspects—including hands on experience using PiML—in a two-session seminar:

**Session 1: Machine Learning Interpretability**

[Download Slides and Jupyter Notebooks](#)

- Post-hoc explainability tools for black box models
  - Local explainability: LIME and SHAP
  - Global explainability: Variable Importance (VI), PDP and ALE
- Limitation and pitfalls of post hoc explainability
- Deep ReLU Networks as Inherehently Interpretable Models
  - Local Linear Model Representation of ReLU DNN and Interpretability
  - Controlling Model Complexity through Regularization
- Functional ANOVA (FANOVA) and Interpretable Model Representation
  - Explainable Boosting Machine
  - Generalized Additive Model with structed Interactions (GAMI) Networks

**Session 2: Machine Learning Model Diagnostics and Validation**

- Model Diagnostics and Testing
  - Weak Spot Analysis through Error Slicing
  - Identification of Over and Underfitting Regions
  - Robustness Testing
  - Reliability (Prediction Uncertainty) Testing using Conformal Prediction
  - Resiliency Testing under Input Distribution Drift
- Model Comparison
  - Arbitrary black box vs. Inherently interpretable models
  - Performance, Robustness and Resilience

