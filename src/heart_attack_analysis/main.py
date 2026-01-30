import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

#Load dataset and basic eda
df=pd.read_csv("/Users/apple/Desktop/python_projects/src/heart_attack_analysis/heart.csv")
describe=df.describe()
# # print(describe)
# df.info()
#  #   Column    Non-Null Count  Dtype  
# ---  ------    --------------  -----  
#  0   age       303 non-null    int64  
#  1   sex       303 non-null    int64  
#  2   cp        303 non-null    int64  
#  3   trtbps    303 non-null    int64  
#  4   chol      303 non-null    int64  
#  5   fbs       303 non-null    int64  
#  6   restecg   303 non-null    int64  
#  7   thalachh  303 non-null    int64  
#  8   exng      303 non-null    int64  
#  9   oldpeak   303 non-null    float64
#  10  slp       303 non-null    int64  
#  11  caa       303 non-null    int64  
#  12  thall     303 non-null    int64  
#  13  output    303 non-null    int64 


## missing value problem
print(df.isnull().sum())
# age         0
# sex         0
# cp          0
# trtbps      0
# chol        0
# fbs         0
# restecg     0
# thalachh    0
# exng        0
# oldpeak     0
# slp         0
# caa         0
# thall       0
# output      0
# dtype: int64