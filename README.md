# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NATHIN R
RegisterNumber: 212222230090
*/
```
```
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
data=pd.read_csv('/content/drive/MyDrive/Dataset-ml/student_scores.csv')
data
data.info()
x=data.iloc[:,:-1].values
print(x)
y=data.iloc[:,:-1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
print(x_train)
print()
print(x_test)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test )
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="red")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,reg.predict(x_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
a=np.array([[10]])
y_pred1=reg.predict(a)
print(y_pred1)
```
## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![image](https://github.com/NathinR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679646/63f72410-cb34-4679-8954-a0257d692c09)
![image](https://github.com/NathinR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679646/d435b464-b3c2-4b52-a738-75c07b9f2b0c)
![image](https://github.com/NathinR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679646/c89037e2-1767-4aca-8266-d5f8bd1013d0)
![image](https://github.com/NathinR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679646/67c2a6eb-0c2c-47da-9bb3-b27988a778a3)
![image](https://github.com/NathinR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679646/ada3b4b1-52d2-497d-8672-b12f79f67890)
![image](https://github.com/NathinR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679646/616622c1-c64f-4eb7-be59-eca86c992e8f)
![image](https://github.com/NathinR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679646/55d6af7e-d9f7-4bf2-bf3c-6e07b25d05f5)
![image](https://github.com/NathinR/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679646/3743e312-4a8b-44a0-a659-72d5687ba631)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
