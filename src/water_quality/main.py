#  Keşifsel veri analizi
import numpy as np  # linear algebra
import pandas as pd # data processing
import seaborn as sns # visualization
import matplotlib.pyplot as plt # visualization
import plotly.express as px # visualization

import missingno as msno #missing value analysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,RepeatedStratifiedKFold,train_test_split
from sklearn.metrics import precision_score,confusion_matrix

from sklearn import tree


df = pd.read_csv("/Users/apple/Desktop/python_projects/src/water_quality/water_potability.csv")
describe=df.describe()# describe metodu ile, count, std,mean,25%,50%,min,max görebiliyosun
# df.info()
#info çıktısı ;
# RangeIndex: 3276 entries, 0 to 3275
# Data columns (total 10 columns):
#  #   Column           Non-Null Count  Dtype  
# ---  ------           --------------  -----  
#  0   ph               2785 non-null   float64
#  1   Hardness         3276 non-null   float64
#  2   Solids           3276 non-null   float64
#  3   Chloramines      3276 non-null   float64
#  4   Sulfate          2495 non-null   float64
#  5   Conductivity     3276 non-null   float64
#  6   Organic_carbon   3276 non-null   float64
#  7   Trihalomethanes  3114 non-null   float64
#  8   Turbidity        3276 non-null   float64
#  9   Potability       3276 non-null   int64  
# dtypes: float64(9), int64(1)


#dependent variable analysis(bağımlı değişken analizi)

# 1. Veriyi hazırlarken index'i sıfırlıyoruz, böylece 'Potability' ve 'count' sütunlarımız oluyor
d = df["Potability"].value_counts().reset_index()

# 2. Plotly kodunda sütun isimlerini buna göre eşliyoruz
fig = px.pie(d, 
             values="count",  # value_counts sonucu oluşan sayı sütunu
             names="Potability", # Sınıf isimlerinin olduğu (0 ve 1) sütun
             hole=0.35, 
             opacity=0.8,
             labels={"count": "Number of Samples", "Potability": "label"})

# Etiketleri manuel vermek isterseniz 'names' kısmını şu şekilde güncelleyebilirsiniz:
fig.update_traces(labels=["Not Potable", "Potable"],textposition="outside",textinfo="percent+label") 

fig.update_layout(title=dict(text="Pie Chart of Potability Feature"))
# fig.show()
fig.write_html("potability_pie_chart.html")

# korelasyon analizi ; features arasında çok korelasyon varssa tüm feature lara ihtiyaç yoktur
sns.clustermap(df.corr(),cmap="vlag",dendrogram_ratio=(0.1,0.2),annot=True,linewidths=0.8,figsize=(10,10))
# plt.show()

# distribution of features
non_potable=df.query("Potability == 0")
potable=df.query("Potability == 1")

plt.figure()#yeni bir tablo , bir grafik tablosu oluşturur grafikler üst üste binmesin diye
for ax,col in enumerate(df.columns[:9]):# ax:index(0,1,2..), col:col name (ph,hardness ..)
    plt.subplot(3,3,ax+1)
    plt.title(col)
    sns.kdeplot(x=non_potable[col],label="Non Potable")# dağılımları çizer kdeplot
    sns.kdeplot(x=potable[col],label="Potable")
    plt.legend()#grafiklerin üstündeki renklerin hangi tag a eşit olduğunu belirtir

plt.tight_layout()#subplotların ototmatik çakışmadan yerlerşmelerini sağlar
# plt.show()


# missing value
msno.matrix(df)
plt.show()



# # Preprocessing: missing value problem, train-test split, normalization


df["ph"] = df["ph"].fillna(df.groupby("Potability")["ph"].transform("mean"))
df["Sulfate"]=df["Sulfate"].fillna(df.groupby("Potability")["Sulfate"].transform("mean"))
df["Trihalomethanes"]=df["Trihalomethanes"].fillna(df.groupby("Potability")["Trihalomethanes"].transform("mean"))
print(df.isnull().sum())

# #train-test split

#independent değerler
X=df.drop('Potability',axis=1).values

#target değerler; potable or non-potable
y=df['Potability'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)#random state ilk random sonra radnom
#min-max normalization 0-1 arasında yapmak istiyoruz
x_train_max=np.max(X_train)
x_train_min=np.min(X_train)
X_train=(X_train - x_train_min)/(x_train_max-x_train_min)
X_test=(X_test - x_train_min)/(x_train_max-x_train_min)


# # Modelling:decision tree and random forest

models=[
    ("DTC",DecisionTreeClassifier(max_depth=3)),
    ("RF",RandomForestClassifier())]

finalResult=[]#score list
cmList=[] #confusion matrix list

for name , model in models:
    model.fit(X_train,y_train) # training

    model_result = model.predict(X_test) # prediction

    score=precision_score(y_test,model_result)
    finalResult.append((name,score))
    cm=confusion_matrix(y_test,model_result)

    cmList.append((name,cm))


print(finalResult)

for name, i in cmList:
    plt.figure()
    sns.heatmap(i,annot=True,linewidths=0.8,fmt=".0f")
    plt.title(name)
    plt.show()


# # Evaluation:decision tree visualization


# # Hpyerparameter tuning:random forest