import numpy as np
import pandas as pd
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml.clustering import *
from pyspark.ml.classification import *
from pyspark.ml import PipelineModel
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import ClusteringEvaluator

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
