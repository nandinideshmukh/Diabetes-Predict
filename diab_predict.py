import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

#uploading the dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")

#to check if there are any missing values in the dataset
sum = data.isnull().sum()
print(f'{sum}\n\n')

print(data.columns)
x = data[['smoking_history','gender']]
y = data['diabetes']

print(f'{data}\n\n')

ordinal = OrdinalEncoder(categories=[['No Info','never','former','not current','current','ever'],['Female','Male','Other']])
ordinal.fit(x)
x_encoded = ordinal.transform(x)

x_train , x_test ,y_train ,y_test = train_test_split(x_encoded , y , test_size=0.2 , random_state=42)

# print(x_encoded)

print(x_train , y_train)

data[['smoking_history_encoded','gender_encoded']] = x_encoded
data.drop(['smoking_history','gender'] , inplace=True , axis=1)
print(data)

#this is for feature scaling of blood glucose level which will help us in gradient descent
print(data['blood_glucose_level'].mean())
max = (data['blood_glucose_level']).max()

data ['new_bgl'] = ((data['blood_glucose_level']-138.05)/max)*100

del data['blood_glucose_level']

print(data)

#this may tell us to use logistic regression and feature selection too
sns.pairplot(data=data,x_vars=['age','hypertension','heart_disease','bmi','HbA1c_level','new_bgl','smoking_history_encoded','gender_encoded'],y_vars=['diabetes'])
plt.show()

#first we'll train the data
lg = LogisticRegression()



#find cost and do gradient descent for minimizing the error


