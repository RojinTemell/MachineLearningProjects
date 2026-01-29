import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 


df=pd.read_csv("/Users/apple/Desktop/python_projects/src/heart_attack_analysis/heart.csv")
describe=df.describe()
print(describe)
df.info()
