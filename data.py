import numpy as np
import pandas as pd
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml import PipelineModel, Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import *
from pyspark.ml.feature import ChiSqSelector
from pyspark.storagelevel import StorageLevel # MEMORY_AND_DISK, DISK_ONLY, OFF_HEAP

from pyspark.ml.clustering import *
from pyspark.ml.classification import *
from pyspark.ml.regression import LinearRegression


"""
To load this file into an environment of choice, do the following

  - In Python console (PySpark console):
    >>> execfile('data.py')

  - In Jupyter Notebook:
    [1] %load data.py
"""

generate_array = udf(lambda mean, std, size: np.random.normal(mean, std, size).tolist(), ArrayType(FloatType()))
generate_random = udf(lambda mean, std: float(np.random.normal(mean, std)), FloatType())
generate_int_random = udf(lambda maxv: int(np.floor(np.random.uniform(maxv))), IntegerType())

def generate_inventory():
  rows = [
    ("Apartment", "A", 50, 4000, 1500, 100, 15, 60, 40),
    ("Apartment", "B", 100, 3000, 800, 70, 20, 50, 25),
    ("Apartment", "C", 70, 1000, 300, 30, 10, 30, 25),
    ("House", "A", 40, 7000, 1600, 400, 120, 60, 40),
    ("House", "B", 30, 6000, 1500, 300, 100, 60, 30),
    ("House", "C", 50, 3000, 900, 250, 80, 70, 40),
    ("WG", "B", 70, 2000, 800, 100, 30, 30, 25),
    ("WG", "C", 30, 1000, 250, 60, 25, 20, 30)
  ]

  df = sc.parallelize(rows).toDF(["type","grade","qty","avgprice","stdprice","avgsize","stdsize","avglat","avglng"])
  df = df.withColumn("size", generate_array(col("avgsize"), col("stdsize"), col("qty")))
  df = df.withColumn("size", explode(col("size")))
  df = df.withColumn("price", generate_random(col("avgprice"), col("stdprice")))
  df = df.withColumn("lat", generate_random(col("avglat"), lit(10)))
  df = df.withColumn("lng", generate_random(col("avglng"), lit(10)))
  df = df.withColumn("seller", generate_int_random(lit(30)))
  return df.select("type","grade","seller","price","size","lat","lng").orderBy(rand())

def generate_seller():
  rows = [(i, i%3+1, np.random.normal(3,2.5)) for i in range(30)]
  df = sc.parallelize(rows).toDF(["seller","deposit","discount"])
  df = df.withColumn("discount", when(col("discount")<1.6, lit(0)).otherwise(col("discount")))
  return df

def generate_vector():
  rows = [
    ("A", ["A1","A2","A3"]),
    ("B", ["B1","B2"]),
    ("C", ["C1","C3"]),
    ("D", [])
  ]
  df = sc.parallelize(rows).toDF(["grade", "sub"])
  return df

def generate_1M():
  affine = udf(lambda x, noise: x*10+(noise*x), FloatType())
  d = 1000. # denominator
  df = sc.parallelize([(0, i.item(), (i/d).item()) for i in np.arange(1,1e6)]).toDF(["a","b","x"])
  df = df.withColumn("noise", generate_random(lit(0.), lit(0.2)))
  df = df.withColumn("c", generate_random(lit(1.), lit(0.33))) # independent column
  df = df.withColumn("y", affine(col("x"), col("noise"))) # as regression target
  df = df.withColumn("z", when(abs(col("noise"))>0.2, lit(1)).otherwise(lit(0))) # as class
  df = df.cache() # .persist(pyspark.StorageLevel.MEMORY_ONLY)
  return df.drop("noise").orderBy(rand())

def generate_struct():
  pass

def example_train_cluster(df):
  # Expected input: inventory
  vec = VectorAssembler(inputCols=["price","size","lat","lng"], outputCol="v")
  kmeans = KMeans(featuresCol="v", predictionCol="pred")

  pipe = Pipeline(stages=[vec, kmeans])
  ev   = ClusteringEvaluator(predictionCol="pred", featuresCol="v", distanceMeasure="cosine") # cosine, squaredEuclidean
  grid = ParamGridBuilder().addGrid(kmeans.k, [3,4,5]).build()
  cv   = TrainValidationSplit(estimator=pipe, estimatorParamMaps=grid, evaluator=ev, trainRatio=0.75)
  model = cv.fit(df)

  return model

def example_train_regression(df):
  # Expected input: 1M
  
  #ChiSqSelector(numTopFeatures=3, featuresCol="v",
  #              outputCol="selectedFeatures", labelCol="z")
  vec  = VectorAssembler(inputCols=["x","c"], outputCol="v")
  reg  = LinearRegression(featuresCol="v", predictionCol="pred", labelCol="y", maxIter=5, regParam=0.1, elasticNetParam=0.5)

  pipe = Pipeline(stages=[vec, reg])
  ev   = RegressionEvaluator(metricName="rmse", predictionCol="pred", labelCol="y") # rmse, mse, r2, mae
  grid = ParamGridBuilder().addGrid(reg.regParam, [0.1, 0.2]).build()
  cv   = CrossValidator(estimator=pipe, estimatorParamMaps=grid, evaluator=ev, numFolds=3)
  cv.setParallelism(3)
  model = cv.fit(df)
  return model

def example_train_class(df):
  pass






