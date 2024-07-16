import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

#uploading the dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")
data2 = pd.read_csv("diabetes (1).csv")

data2.drop(['Age'],inplace=True,axis=1)
data2.drop(['BMI'],inplace=True,axis=1)
data2.drop(['Glucose'],inplace=True,axis=1)


#to check if there are any missing values in the dataset
sum = data.isnull().sum()
print(f'{sum}\n\n')

print(data.columns)
x = data[['smoking_history','gender']]
y = data['diabetes']

# print(f'{data}\n\n')

#ordinal encoding used for converting categorical values to numerical values
ordinal = OrdinalEncoder(categories=[['No Info','never','former','not current','current','ever'],['Female','Male','Other']])
ordinal.fit(x)
x_encoded = ordinal.transform(x)

x_train , x_test ,y_train ,y_test = train_test_split(x_encoded , y , test_size=0.2 , random_state=42)

# print(x_encoded)

print(x_train , y_train)

data[['smoking_history_encoded','gender_encoded']] = x_encoded
data.drop(['smoking_history','gender'] , inplace=True , axis=1)
# print(data)

#this is for feature scaling of blood glucose level which will help us in gradient descent
print(data['blood_glucose_level'].mean())
max = (data['blood_glucose_level']).max()

data ['new_bgl'] = ((data['blood_glucose_level']-138.05)/max)*100

del data['blood_glucose_level']

print(data)
print(data2)

#find cost function and do gradient descent for minimizing the error
E = np.array(data['HbA1c_level'])
A= np.array(data['age'])
B= np.array(data['bmi'])
C = np.array(data['hypertension'])
D= np.array([data['new_bgl']])

Y = np.array(data['diabetes'])

def sigmoid(z):
    return 1/(1+np.exp(-z))

#conactenating data
data3 = pd.concat([data , data2 ], axis=1)
data3.drop(['Outcome'],inplace=True,axis=1)
data3.drop(['BloodPressure'],inplace=True,axis=1)
print(data3)

#identifying missing values
print(data3.isna().sum())

#filling missing values
handled_data = data3.fillna({'Pregnancies':data3['Pregnancies'].mean()
                             ,'DiabetesPedigreeFunction':data3['DiabetesPedigreeFunction'].mean(),
                             'Insulin':0,'SkinThickness':0})

#if missing values are filled or not
print(handled_data.isna().sum())

#useful feature-selection
#pregnancy is a major feature for this
sns.heatmap(data2.corr(),annot=True)
plt.show()

sns.heatmap(handled_data.corr(),annot=True)
plt.show()

# #this may tell us to use logistic regression and feature selection too
sns.pairplot(data=handled_data,x=['age','hypertension','heart_disease','bmi','HbA1c_level','new_bgl','smoking_history_encoded','gender_encoded','Pregnancies','SkinThickness','Insulin','DiabetesPedigreeFunction'],y=['diabetes'],kde=True)
plt.show()

#next we'll train the data


