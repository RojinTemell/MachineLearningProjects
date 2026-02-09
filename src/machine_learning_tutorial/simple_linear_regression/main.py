
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

df=pd.read_csv("/Users/apple/Desktop/python_projects/src/machine_learning_tutorial/simple_linear_regression/Salary_dataset.csv")
# des=df.describe()
# print(des)
# head=df.head()
# print(head)

X=df["YearsExperience"].values.reshape(-1,1)
y=df["Salary"].values

plt.plot(X, y, marker="o")
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
# plt.show()

# train and test datas
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=42)
model =LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

#Calculate metrics
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)
mae =mean_absolute_error(y_test,y_pred)
# Print metrics
print("=" * 50)
print("MODEL EVALUATION METRICS")
print("=" * 50)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print("=" * 50)
print(f"\nModel Equation: Salary = {model.coef_[0]:.2f} * YearsExperience + {model.intercept_:.2f}")
print("=" * 50)
# results_df = pd.DataFrame({
#     "y_actual":y_test,
#     "y_predict":y_pred
# })
# print(results_df)
plt.figure()
# plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, s=100)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted')
plt.grid(True)
# plt.show()