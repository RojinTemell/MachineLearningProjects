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
describe=df.describe()
# df.info()

#dependent variable analysis(bağımlı değişken analizi)
d = df["Potability"].value_counts().reset_index()
d["Potability"] = d["Potability"].map({
    0: "Not Potable",
    1: "Potable"
})
d.columns = ["Potability", "count"]
fig=px.pie(d,values="count",names="Potability",hole=0.35,opacity=0.8,
           labels={
               "Potability":"label","count":"Number of Samples"
           })
fig.update_layout(title=dict(text="Pie Chart of Potability Feature"))
fig.update_traces(textposition="outside",textinfo="percent+label")
# fig.show()
fig.write_html("potability_pie_chart.html")

# korelasyon analizi ; features arasında çok korelasyon varssa tüm feature lara ihtiyaç yoktur
sns.clustermap(df.corr(),cmap="vlag",dendrogram_ratio=(0.1,0.2),annot=True,linewidths=0.8,figsize=(10,10))
# plt.show()

# distribution of features
non_potable=df.query("Potability == 0")
potable=df.query("Potability == 1")

plt.figure()
for ax,col in enumerate(df.columns[:9]):
    plt.subplot(3,3,ax+1)
    plt.title(col)
    sns.kdeplot(x=non_potable[col],label="Non Potable")
    sns.kdeplot(x=potable[col],label="Potable")
    plt.legend()

plt.tight_layout()
# plt.show()
# missing value
msno.matrix(df)
plt.show()



# Preprocessing: missing value problem, train-test split, normalization



# Modelling:decision tree and random forest


# Evaluation:decision tree visualization


# Hpyerparameter tuning:random forest