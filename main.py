# Problem Statement : Predict house price using features like income, rooms, location, etc.

# Step 1: Import Libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

# Step 2: Load Dataset
df = pd.read_csv("housing.csv")
df.head()

# Step 3: Understand Data
df.info()
df.describe()
df.isnull().sum()
df.columns

# Step 4: Data Cleaning
# Remove Missing Values
df.dropna(inplace=True)

# Step 5: Encode Categorical Data
df = pd.get_dummies(df, drop_first=True)

# Step 6: Exploratory Data Analysis (EDA)
# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
plt.show()

# Pairplot 
sns.pairplot(df)
plt.show()

# Step 7: Define Features & Target
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Step 8: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# Step 9: Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 10: Train Models
# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
plt.scatter(y_test, lr_pred)
plt.show()

# Decision Tree
dt = DecisionTreeRegressor(max_depth=5)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
plt.scatter(y_test, dt_pred)
plt.show()

# Random Forest
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
plt.scatter(y_test, rf_pred)
plt.show()

# Step 9: Evaluate Models
def evaluate(y_test, pred):
     print("MAE:", mean_absolute_error(y_test, pred))
     print("MSE:", mean_squared_error(y_test, pred))
     print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))

# Results
print("Linear:", evaluate(y_test, lr_pred))
print("Decision Tree:", evaluate(y_test, dt_pred))
print("Random Forest:", evaluate(y_test, rf_pred))

# Check average price
mean_val = y.mean()
mean_val

Linear_Error = (np.sqrt(mean_squared_error(y_test, lr_pred))/ mean_val)*100
print(Linear_Error)

Decision_Tree_Error = (np.sqrt(mean_squared_error(y_test, dt_pred))/ mean_val)*100
print(Decision_Tree_Error)

Random_Forest_Error = (np.sqrt(mean_squared_error(y_test, rf_pred))/ mean_val)*100
print(Random_Forest_Error)

# R² Score
r2_score(y_test, lr_pred)
r2_score(y_test, dt_pred)
r2_score(y_test, rf_pred)