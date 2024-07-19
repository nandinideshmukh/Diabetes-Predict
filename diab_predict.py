import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_curve,auc,accuracy_score

#uploading the dataset
data = pd.read_csv("diabetes_prediction_dataset.csv")
data2 = pd.read_csv("diabetes (1).csv")

# data2.drop(['Age'],inplace=True,axis=1)
data2.drop(['Outcome'],inplace=True,axis=1)
data2.drop(['BMI'],inplace=True,axis=1)
data2.drop(['DiabetesPedigreeFunction'],inplace=True,axis=1)
#data2.drop(['Glucose'],inplace=True,axis=1)

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

# print(x_encoded)

data[['smoking_history_encoded','gender_encoded']] = x_encoded
data.drop(['smoking_history','gender'] , inplace=True , axis=1)
data2.drop(['BloodPressure'],inplace=True,axis=1)

# print(data)

#this is for feature scaling of blood glucose level which will help us in gradient descent
# print(data['blood_glucose_level'].mean())
# max = (data['blood_glucose_level']).max()

# data ['new_bgl'] = ((data['blood_glucose_level']-138.05)/max)*100

# del data['blood_glucose_level']

print(data.shape)
print(data2.shape)

#merging data

# data2.rename(columns={'BMI':'bmi'},inplace=True)
data2.rename(columns={'Glucose':'blood_glucose_level'},inplace=True)

data3 = pd.merge(data,data2,on=['blood_glucose_level'],how='inner')
data3.drop(['Age'],inplace=True,axis=1)
# data3.drop(['DiabetesPedigreeFunction'],inplace=True,axis=1)

# data3.drop(['Outcome'],inplace=True,axis=1)
data3.drop(['SkinThickness'],inplace=True,axis=1)
print("Merged data is:",data3)

#identifying missing values
print(data3.isna().sum())

#filling missing values
handled_data = data3.fillna({'Pregnancies':data3['Pregnancies'].mean(),
                             'Insulin':data3['Insulin'].mean()})

handled_data = data3.fillna({'blood_glucose_level':data3['blood_glucose_level'].mode(),'heart_disease ':data3['heart_disease'].mean()})

print("Handled data is: ",handled_data)

#if missing values are filled or not
print("Number of null values in handled data is: ",handled_data.isna().sum())

#useful feature-selection
#pregnancy is a major feature for this
sns.heatmap(data2.corr(),annot=True)
plt.title("Data 2 co -relation")
plt.show()

sns.heatmap(handled_data.corr(),annot=True)
plt.title("Data 1 co-relation")
plt.show()

# #this may tell us to use logistic regression and feature selection too
# sns.pairplot(data=handled_data,x=['age','hypertension','heart_disease','bmi','HbA1c_level','new_bgl','smoking_history_encoded','gender_encoded','Pregnancies','Insulin','DiabetesPedigreeFunction'],y=['diabetes'],kde=True)
# plt.show()

#parameters that affect the count too much can be given more importance in the data
#use different plots so that there is no overlapping and better understanding
#based on these results , elimination of features

columns_to_plot = ['age','Hypertension','heart_disease','bmi','HbA1c_level','blood_glucose_level','smoking_history_encoded','gender_encoded','Pregnancies','Insulin']

# for columns in (columns_to_plot):
#     sns.histplot(handled_data[columns] , kde=True)
#     plt.title(f'Histogram of {columns}')
#     plt.show()

print("Checking if data is empty or not: ",handled_data.isna().sum())

# Check for infinite values
print(np.isinf(handled_data).sum())

print(handled_data.shape)

#next we'll train the data
target = handled_data['diabetes']
X_train ,X_test , Y_train,Y_test = train_test_split(handled_data.drop('diabetes', axis=1),target,test_size=0.3)

#find the best fit parameters for RandomForestClassifier
# estimators = [10,25,40,60,100,125]

# param_grid = {
#    # 'n_estimators': [100, 200, 300],
#    # 'max_depth': [None, 10, 20, 30],
#    #   'min_samples_split': [2, 5, 10],
#    # 'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt', 'log2']
#    }

# fitting = GridSearchCV(RandomForestClassifier(random_state=42),param_grid,cv=10,n_jobs=2,verbose=2)

rc = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_leaf=1,min_samples_split=3)
rc.fit(X_train,Y_train)
# fitted = fitting.best_estimator_

pred1 = rc.predict_proba(X_test)[:, 1]

fpr1 ,tpr1 ,thresholds = roc_curve(Y_test,pred1)

plt.figure()
plt.plot(fpr1,tpr1)
plt.fill_between(fpr1, tpr1, color='black', alpha=0.4)
plt.xlim([0,1.05])
plt.ylim([0,1.05])
plt.plot([0,1],[1,0],color = 'black')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.show()


# area2 = auc(fpr2,tpr2)
area1 = auc(fpr1,tpr1)
print("Probability of positive class is: ",area1*100)
# print("Probability of negative class is: ",area2*100)


gender = int(input('Enter your gender\n 0:Female\n 1: Male\n 2:Other: '))
smoking = int(input('Enter your soming hsitory\n 0:No Info\n1:never\n2:former\n3:not current\n4: current\n5:ever'))
age = int(input("Enter your age: "))
hypertension = int(input("Do you have hypertension(enter 1 if yes else enter 0): "))
heartdisease = int(input("Do you have heartdisease(enter 1 if yes else enter 0): "))
bmi = float(input("Enter your bmi: "))
Haemoglobin = float(input("Enter you Hb: "))
glucose = int(input("Enter your bllood_glucose_level"))
pregnancy = int(input("How many pregnancies you had? : "))
insulin = int(input("Enter your insulin level: "))

testing_data = pd.DataFrame([[age,hypertension,heartdisease, bmi, Haemoglobin, glucose,smoking, gender,  pregnancy, insulin]],
                        columns=['age','hypertension','heart_disease','bmi','HbA1c_level', 'blood_glucose_level', 'smoking_history_encoded', 'gender_encoded','Pregnancies', 'Insulin'])

tested_data = rc.predict_proba(testing_data)[:,1]
if tested_data[0]*100<50:
      print("Person is not diabetic")
else:
      print("Person is diabetic")
print("Prediction that the person is diabetic is: ",tested_data[0]*100)


#Data is handled properly and task is completed
