# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset and get data info
2. Check for null values
3. Map values for position column
4. Split the dataset into train and test set
5. Import decision tree regressor and fit it for data
6. Calculate MSE value, R2 value and Data predict.

## Program:
```python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Udayakumar R
RegisterNumber:  212222230163
*/

import pandas as pd
data = pd.read_csv('Salary.csv')
data.head()
data.info()
data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x = data[["Position","Level"]]
y = data["Salary"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2 = metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output :
## DATA HEAD:
![image](https://github.com/R-Udayakumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708024/ac5774e9-fc9b-46bb-8db1-1e517e80d5c6)
## DATA INFO:
![image](https://github.com/R-Udayakumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708024/95f4ca7a-0ac0-4979-a3b2-4802376e4dba)
## NULL VALUE :
![image](https://github.com/R-Udayakumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708024/ea99ba2f-685b-455f-8b60-ba75c5426c29)
## DATA HEAD AFTER LABEL ENCODER :
![image](https://github.com/R-Udayakumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708024/a5c21d73-2d67-481f-8194-715009b0c448)
## MSE VALUE :
![image](https://github.com/R-Udayakumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708024/f7079522-e1b7-4d7a-a2f2-46f4fd7ecad6)

## R2 VALUE :
![image](https://github.com/R-Udayakumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708024/975f0e03-f8b8-4ffc-8c03-46dbdfc96831)

## DATA PREDICTION :
![image](https://github.com/R-Udayakumar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708024/bcb432b9-7454-4b17-a207-cdef7efb1c0b)




## Result :
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
