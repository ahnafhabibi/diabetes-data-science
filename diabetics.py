## Libraries
import prince
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns
import imblearn
import random
import sklearn
import mixed_naive_bayes
import joblib

diabetics=pd.read_csv("diabetes_data_upload.csv")
female=diabetics[diabetics["Gender"]=='Female']
male=diabetics[diabetics["Gender"]=='Male']
from sklearn.model_selection import train_test_split
female_train, female_test = train_test_split(female, test_size=0.15,random_state=1)
from sklearn.model_selection import train_test_split
male_train, male_test = train_test_split(male, test_size=0.10,random_state=1)
diabetics_train=pd.concat([male_train,female_train])
diabetics_train=diabetics_train.reset_index()
diabetics_test=pd.concat([male_test,female_test])
diabetics_test=diabetics_test.reset_index()
diabetics_train=diabetics_train.replace(to_replace="Yes",value=1)
diabetics_train=diabetics_train.replace(to_replace="No",value=0)
diabetics_train=diabetics_train.replace(to_replace="Male",value=1)
diabetics_train=diabetics_train.replace(to_replace="Female",value=0)
diabetics_train=diabetics_train.replace(to_replace="Positive",value=1)
diabetics_train=diabetics_train.replace(to_replace="Negative",value=0)
female_train=diabetics_train[diabetics_train['Gender']==0]
female_train=female_train.reset_index()
female_train=female_train.loc[:,female_train.columns!='index']
female_train=female_train.loc[:,female_train.columns!='level_0']
# import library
from imblearn.over_sampling import SMOTENC
smote = SMOTENC(categorical_features=list(range(2,16)))
# fit predictor and target variable
x=female_train.loc[:,female_train.columns!='class']
y=female_train["class"]
x_smote, y_smote = smote.fit_resample(x, y)
print(x_smote.head())
x_smote=x_smote.replace(to_replace=1,value="Yes")
x_smote=x_smote.replace(to_replace=0,value="No")
y_smote=y_smote.replace(to_replace=1,value="Positive")
y_smote=y_smote.replace(to_replace=0,value="Negative")
x_smote['class']=y_smote
female_train=x_smote.copy()
diabetics_train=pd.concat([male_train,female_train])
diabetics_train=diabetics_train.replace(to_replace="Yes",value=1)
diabetics_train=diabetics_train.replace(to_replace="No",value=0)
diabetics_train=diabetics_train.replace(to_replace="Male",value=1)
diabetics_train=diabetics_train.replace(to_replace="Female",value=0)
diabetics_train=diabetics_train.replace(to_replace="Positive",value=1)
diabetics_train=diabetics_train.replace(to_replace="Negative",value=0)
from mixed_naive_bayes import MixedNB
X=diabetics_train.loc[:,diabetics_train.columns!='class']
y=diabetics_train["class"]
clf=MixedNB(categorical_features=[1,15])
clf.fit(X,y)
diabetics_test=diabetics_test.replace(to_replace="Yes",value=1)
diabetics_test=diabetics_test.replace(to_replace="No",value=0)
diabetics_test=diabetics_test.replace(to_replace="Male",value=1)
diabetics_test=diabetics_test.replace(to_replace="Female",value=0)
diabetics_test=diabetics_test.replace(to_replace="Positive",value=1)
diabetics_test=diabetics_test.replace(to_replace="Negative",value=0)
diabetics_test=diabetics_test.loc[:,diabetics_test.columns!='index']
X=diabetics_test.loc[:,diabetics_test.columns!='class']
y=diabetics_test["class"]
pred_values=clf.predict(X)
y_actu = pd.Series(y, name='Actual')
y_pred = pd.Series(pred_values, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
print(df_confusion)
X['actual']=y
X['pred']=pred_values
joblib.dump(clf, 'fhs_clf_model.pkl')