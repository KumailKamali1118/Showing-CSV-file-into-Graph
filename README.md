import numpy as np
import pandas as pd
data=pd.read_csv(r'C:\Users\Gulzar Hussain\Desktop\AI-Data.csv') 
data.info()
data.head()
data.describe()
%matplotlib inline
import matplotlib.pyplot as plt
data.hist(bins=10,figsize=(15,25))
print('Percentage',data.PlaceofBirth.value_counts(normalize=True))
data.PlaceofBirth.value_counts(normalize=True).plot(kind='bar')
print('Percentage',data.NationalITy.value_counts(normalize=True))
data.NationalITy.value_counts(normalize=True).plot(kind='pie')print('Percentage',data.Semester.value_counts(normalize=True))
data.Semester.value_counts(normalize=True).plot(kind='pie')
print('Percentage',data.raisedhands.value_counts(normalize=True))
data.raisedhands.value_counts(normalize=True).plot(kind='pie',figsize=(30,40),fontsize=(25))
print('Percentage',data.Class.value_counts(normalize=True))
data.Class.value_counts(normalize=True).plot(kind='bar')
print(data.shape)
data.columns
print('Percentage',data.gender.value_counts(normalize=True))
data.gender.value_counts(normalize=True).plot(kind='pie')
print('Percentage',data.StageID.value_counts(normalize=True))
data.StageID.value_counts(normalize=True).plot(kind='bar')
print('Percentage',data.Topic.value_counts(normalize=True))
data.Topic.value_counts(normalize=True).plot(kind='pie')
print('Percentage',data.ParentAnsweringSurvey.value_counts(normalize=True))
data.ParentAnsweringSurvey.value_counts(normalize=True).plot(kind='bar')
import matplotlib.pyplot as plt
plt.scatter(data['raisedhands'],data['VisITedResources'])
print('Percentage',data.StudentAbsenceDays.value_counts(normalize=True))
data.StudentAbsenceDays.value_counts(normalize=True).plot(kind='pie')
data['StudentAbsenceDays'].value_counts()
data['Class'].value_counts()
print('Percentage',data.Class.value_counts(normalize=True))
data.Class.value_counts(normalize=True).plot(kind='pie')
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
fig,axarr= plt.subplots(2,2,figsize=(10,10))
sns.countplot(x='Class',data=data, ax=axarr[0,0])
sns.countplot(x='gender',data=data, ax=axarr[0,1])
sns.countplot(x='VisITedResources',data=data, ax=axarr[1,0])
sns.countplot(x='VisITedResources',data=data, ax=axarr[1,1])
fig,(axis1,axis2)=plt.subplots(1,2,figsize=(10,10))
sns.swarmplot(x='gender',y='raisedhands',data=data,ax=axis1)
sns.swarmplot(x='gender',y='AnnouncementsView',data=data,ax=axis2)
sns.pairplot(data,hue='Class')
Features= data.drop('gender',axis=1)
Target = data['gender']
label= LabelEncoder()
Cat_Colums=Features.dtypes.pipe(lambda Features:Features[Features=='object']).index
for col in Cat_Colums:
    Features[col]=label.fit_transform(Features[col])
    print (Features)
Features= data.drop('raisedhands',axis=1)
Target = data['raisedhands']
label= LabelEncoder()
Cat_Colums=Features.dtypes.pipe(lambda Features:Features[Features=='object']).index
for col in Cat_Colums:
    Features[col]=label.fit_transform(Features[col])
    print (Features)
Features= data.drop('raisedhands',axis=1)
Target = data['raisedhands']
label= LabelEncoder()
Cat_Colums=Features.dtypes.pipe(lambda Features:Features[Features=='object']).index
for col in Cat_Colums:
    Features[col]=label.fit_transform(Features[col])
    print (Features)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
x_train,x_test,y_train,y_test=train_test_split(Features,Target,test_size=0.2,random_state=52)
print(x_train)
print(x_test)
print(y_train)
print(y_test)
Logit_Model=LogisticRegression()
Logit_Model.fit(x_train,y_train)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
Prediction=Logit_Model.predict(x_test)
score = accuracy_score(y_test,Prediction)
Report=classification_report(y_test,Prediction)
Prediction
