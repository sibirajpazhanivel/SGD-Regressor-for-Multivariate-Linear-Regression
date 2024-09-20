# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: MANOJ MV
RegisterNumber:  212222220023
*/
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())

X = df.drop(columns=['AveOccup','HousingPrice'])
X.info()

Y = df[['AveOccup','HousingPrice']]
Y.info()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)

Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)

print("\nPredictions:\n", Y_pred[:5])
```

## Output:
![image](https://github.com/user-attachments/assets/8e3d724b-5f63-41f7-9c04-c4f6566fc1c4)

![image](https://github.com/user-attachments/assets/a4b69b32-cdd0-41a1-a86d-cbe47010776d)

![image](https://github.com/user-attachments/assets/199b8564-1f28-4807-a5e9-90858d7faf5f)

![image](https://github.com/user-attachments/assets/729262b5-26e5-48a3-aef6-19ac7c2178b2)

![image](https://github.com/user-attachments/assets/eada08ef-8d4e-4453-b278-83f73c85d9af)

![image](https://github.com/user-attachments/assets/c78783f7-f656-487f-9c8e-d19cc8bdea39)
## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
