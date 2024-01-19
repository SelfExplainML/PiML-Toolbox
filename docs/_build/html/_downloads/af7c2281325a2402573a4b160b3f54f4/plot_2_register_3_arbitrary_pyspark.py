# -*- coding: utf-8 -*-
"""
Register PySpark Models
========================================================
Here we show how to write a wrapper for PySpark models.
"""

#%%
# For demonstration, we first fit a Decision tree model using SimuCredit data.
import os
os.environ["PYSPARK_PYTHON"] = "python"

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer

data = pd.read_csv('https://github.com/SelfExplainML/PiML-Toolbox/blob/main/datasets/SimuCredit.csv?raw=true')
feature_names = ['Mortgage', 'Balance', 'Amount Past Due', 'Credit Inquiry', 'Open Trade', 'Delinquency', 'Utilization']
target_name = 'Approved'

spark = SparkSession.builder.appName("SimuCredit-Spark-Demo").getOrCreate()
spark_df = spark.createDataFrame(data)

feature_assembler = VectorAssembler(inputCols=feature_names, outputCol="features")
label_stringIdx = StringIndexer(inputCol=target_name, outputCol='label')

pipeline = Pipeline(stages=[feature_assembler, label_stringIdx])
pipelineModel = pipeline.fit(spark_df)
train_data, test_data = pipelineModel.transform(spark_df).randomSplit([0.8, 0.2], seed=2024)

dt = DecisionTreeClassifier(featuresCol='features', labelCol='label', seed=2024)
model = dt.fit(train_data)

#%%
# Next, we define the wrapper functions of predict and predict_proba.

def predict_func(X):

    spark_df = pipelineModel.transform(spark.createDataFrame(pd.DataFrame(X, columns=feature_names)))
    pred = model.transform(spark_df).select('prediction').toPandas().values.astype(float).ravel()
    return pred

def predict_proba_func(X):
    
    spark_df = pipelineModel.transform(spark.createDataFrame(pd.DataFrame(X, columns=feature_names)))
    predictions = model.transform(spark_df).select('probability').toPandas()
    proba = predictions.explode('probability').values.reshape((-1, 2)).astype(float)
    return proba

#%%
# Register the fitted model into PiML (please make sure the datasets of different pipelines are the same)
from piml import Experiment
exp = Experiment(highcode_only=True)
pipeline = exp.make_pipeline(predict_func=predict_func,
                             predict_proba_func=predict_proba_func,
                             task_type="classification",
                             train_x=train_data.select(feature_names).toPandas().values,
                             train_y=train_data.select(target_name).toPandas().values,
                             test_x=test_data.select(feature_names).toPandas().values,
                             test_y=test_data.select(target_name).toPandas().values,
                             feature_names=feature_names,
                             target_name=target_name)
exp.register(pipeline, "Spark-GBT")

#%%
# Check model performance
exp.model_diagnose(model="Spark-GBT", show="accuracy_table")

#%%
# Run validataion tests
exp.model_explain(model="Spark-GBT", show="pdp", uni_feature="Balance", sample_size=1000, figsize=(5, 4))
