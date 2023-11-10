# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary packages using import statement.
2. Read the given csv file using read_csv() method and print the number of contents to be 
   displayed using df.head().
3. import KMeans and use for loop to cluster the data.
4. Predict the cluster and plot data graphs.
5.Print the outputs and end the program

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SANDHYA BN
RegisterNumber:  212222040144
import pandas as pd
import numpy as np
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
data["salary"]=l.fit_transform(data["salary"])
data.head()
data["Departments "]=l.fit_transform(data["Departments "])
data.head()
data.info()
data.shape
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','Departments ','salary']]
x.head()
x.shape
x.info()
y=data['left']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy = ",accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2,1]])
*/
```

## Output:

## Read csv file:

![1st](https://github.com/sandhyabalamurali/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115525118/1fa81678-a443-4492-90f5-bfd9e2ac7e3c)

## Dataset info:

![2nd](https://github.com/sandhyabalamurali/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115525118/8dc98e18-93fc-42d3-9e60-ea09fc1d57c6)


![3rd](https://github.com/sandhyabalamurali/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115525118/28c165b9-3c62-49f6-a3e9-63ac151db6dd)

## Dataset Value count:

![4th](https://github.com/sandhyabalamurali/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115525118/4e7b813b-00e5-41f5-a3d8-e47ee191fcde)

## Dataset head:

![5th](https://github.com/sandhyabalamurali/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115525118/c3623002-779a-434f-903b-0410d47c67d6)

## Data info:

![6th](https://github.com/sandhyabalamurali/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115525118/2f2af6b5-62ac-4ec5-bcd1-ad5addc49fba)

## Dataset shape:

![7th](https://github.com/sandhyabalamurali/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115525118/b1855057-332a-46d6-bb5f-9cba85343625)


![8th](https://github.com/sandhyabalamurali/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115525118/fb717cb9-2c8f-49f8-9937-09f7a35952a0)

## Y-Pred:


![9th](https://github.com/sandhyabalamurali/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115525118/796b1cc0-723a-4299-82af-2b7f10a706cb)

## Accuracy:

![iiii](https://github.com/sandhyabalamurali/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115525118/253191eb-e251-4c2b-861f-61c100e596b7)


## Dataset Predict:

![10th](https://github.com/sandhyabalamurali/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/115525118/618a35b7-dda8-49c7-90d5-8e705d97b90e)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
