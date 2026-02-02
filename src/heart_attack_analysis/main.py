import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler

#Load dataset and basic eda
df=pd.read_csv("/Users/apple/Desktop/python_projects/src/heart_attack_analysis/heart.csv")
# describe=df.describe()
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


# categorical and numerical feature analysis
categorical_list=["sex" ,"cp","fbs","restecg","exng","slp","caa","thall","output"]
df_categorical=df.loc[:,categorical_list] 
# for i in categorical_list:
#     plt.figure()
#     sns.countplot(x=i,data=df_categorical,hue = "output")
#     plt.title(i)
# plt.show()

numerical_list=["age","trtbps","chol","thalachh","oldpeak","output"]
df_numerical=df.loc[:,numerical_list]
# sns.pairplot(df_numerical,hue = "output",diag_kind="kde")
# plt.show()

#EDA:box, swarm ,cat,correlation analysis

scaler=StandardScaler()
scaled_array=scaler.fit_transform(df[numerical_list[:-1]])

df_dummy=pd.DataFrame(scaled_array,columns=numerical_list[:-1])
df_dummy=pd.concat([df_dummy,df.loc[:,"output"]],axis=1)

#box plot 

data_melted=pd.melt(df_dummy,id_vars="output",var_name="features",value_name="value")
# plt.figure()
# sns.boxplot(x="features",y="value",hue="output",data=data_melted)
# plt.show()

#sworm plot
# plt.figure()
# sns.swarmplot(x="features",y="value",hue="output",data=data_melted)
# plt.show()

#cat plot
# plt.figure()
# # sns.catplot(x="chol",y="sex", col="age"  ,kind="swarm",hue="output",data=df)
# plt.show()

#correlation
plt.figure()
sns.heatmap(df.corr(),annot=True,fmt=".1f",linewidths=0.7)
# plt.show()


# outlier detection

for i in numerical_list:
    Q1=np.percentile(df.loc[:,i],25)
    Q3=np.percentile(df.loc[:,i],75)

    IQR=Q3 -Q1
    print(f" {i}old shape :{df.loc[:,i].shape}")
    upper=np.where(df.loc[:,i]>=(Q3+2.5*IQR))
    lower=np.where(df.loc[:,i]<=(Q1-2.5*IQR))

    try:
        df.drop(upper[0],inplace=True)
    except: print("hata")

    try:
        df.drop(lower[0],inplace=True)
    except: print("hata")

    print(f"New shape :{df.shape}")


