# -*- coding: utf-8 -*-
"""
Created on Sat May 25 10:25:07 2019

@author: gthangar
"""
import pandas as pd
#import re
import nltk
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#import dataframes as df
nltk.download('stopwords')
#import pickle;
import numpy as np
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
documents=[]
stemmer=WordNetLemmatizer()
train_data=pd.read_csv("./train.csv")
test_data=pd.read_csv("./test.csv")
print(train_data.describe())
sns.barplot(x=train_data["PAN_flag"],y=train_data["VoterID_flag"])
print(train_data.head())
print(set(train_data["Employment.Type"]))
train_data=train_data.fillna({"Employment.Type":"O"})
test_data=test_data.fillna({"Employment.Type":"O"})
print(set(train_data["Employment.Type"]))
print(set(test_data["Employment.Type"]))
Employment_mapping={"Salaried":0,"Self employed":1,"O":2}
train_data["Employment.Type"]=train_data["Employment.Type"].map(Employment_mapping)
test_data["Employment.Type"]=test_data["Employment.Type"].map(Employment_mapping)
print(set(train_data["Employment.Type"]))
print(set(train_data["AVERAGE.ACCT.AGE"]))

def parse_dates(date):
    date=str(date)
    date=date.split()
    year=date[0][0]
    year=int(year)
    Month=date[1][0]
    Month=int(Month)
    #print(year)
    #print(Month)
    total_month=year+Month/12
    #print(total_month)
    return total_month
train_data["AVERAGE.ACCT.AGE"]=train_data["AVERAGE.ACCT.AGE"].apply(lambda row:parse_dates(row))

def parse_dates1(date):
    date=str(date)
    date=date.split()
    year=date[0][0]
    year=int(year)
    Month=date[1][0]
    Month=int(Month)
    #print(year)
    #print(Month)
    total_month=year+Month/12
    #print(total_month)
    return total_month
train_data["CREDIT.HISTORY.LENGTH"]=train_data["CREDIT.HISTORY.LENGTH"].apply(lambda row:parse_dates1(row))
'''
def parse_dates2(date):
    date=str(date)
    date=date.split()
    year=date[0][0]
    year=int(year)
    Month=date[1][0]
    Month=int(Month)
    #print(year)
    #print(Month)
    total_month=year+Month/12
    #print(total_month)
    return total_month
test_data["AVERAGE.ACCT.AGE"]=test_data["AVERAGE.ACCT.AGE"].apply(lambda row:parse_dates2(row))

def parse_dates3(date):
    date=str(date)
    date=date.split()
    year=date[0][0]
    year=int(year)
    Month=date[1][0]
    Month=int(Month)
    #print(year)
    #print(Month)
    total_month=year+Month/12
    #print(total_month)
    return total_month
test_data["AVERAGE.ACCT.AGE"]=test_data["AVERAGE.ACCT.AGE"].apply(lambda row:parse_dates3(row))
'''

print(train_data["AVERAGE.ACCT.AGE"])
'''
scale1=StandardScaler()
scale1.fit_transform(test_data[["UniqueID","asset_cost","Employment.Type","PERFORM_CNS.SCORE","PRI.NO.OF.ACCTS","PRI.ACTIVE.ACCTS","PERFORM_CNS.SCORE","PRI.NO.OF.ACCTS","SEC.OVERDUE.ACCTS","SEC.CURRENT.BALANCE","SEC.SANCTIONED.AMOUNT","SEC.DISBURSED.AMOUNT","NEW.ACCTS.IN.LAST.SIX.MONTHS","DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS","NO.OF_INQUIRIES","AVERAGE.ACCT.AGE","CREDIT.HISTORY.LENGTH"]].values)
XX_Train=test_data[["UniqueID","asset_cost","Employment.Type","PERFORM_CNS.SCORE","PRI.NO.OF.ACCTS","PRI.ACTIVE.ACCTS","PERFORM_CNS.SCORE","PRI.NO.OF.ACCTS","SEC.OVERDUE.ACCTS","SEC.CURRENT.BALANCE","SEC.SANCTIONED.AMOUNT","SEC.DISBURSED.AMOUNT","NEW.ACCTS.IN.LAST.SIX.MONTHS","DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS","NO.OF_INQUIRIES","AVERAGE.ACCT.AGE","CREDIT.HISTORY.LENGTH"]]
XX_Train,XX_val,YY_val=train_test_split(XX_Train,test_size=0.1)
adaboost=AdaBoostClassifier()
adaboost.fit(XX_Train,)
y1_pred=LogReg.predict(XX_val)
'''
scale=StandardScaler()
scale.fit_transform(train_data[["UniqueID","asset_cost","Employment.Type","PERFORM_CNS.SCORE","PRI.NO.OF.ACCTS","PRI.ACTIVE.ACCTS","PERFORM_CNS.SCORE","PRI.NO.OF.ACCTS","SEC.OVERDUE.ACCTS","SEC.CURRENT.BALANCE","SEC.SANCTIONED.AMOUNT","SEC.DISBURSED.AMOUNT","NEW.ACCTS.IN.LAST.SIX.MONTHS","DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS","NO.OF_INQUIRIES","AVERAGE.ACCT.AGE","CREDIT.HISTORY.LENGTH"]].values)
X_Train=train_data[["UniqueID","asset_cost","Employment.Type","PERFORM_CNS.SCORE","PRI.NO.OF.ACCTS","PRI.ACTIVE.ACCTS","PERFORM_CNS.SCORE","PRI.NO.OF.ACCTS","SEC.OVERDUE.ACCTS","SEC.CURRENT.BALANCE","SEC.SANCTIONED.AMOUNT","SEC.DISBURSED.AMOUNT","NEW.ACCTS.IN.LAST.SIX.MONTHS","DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS","NO.OF_INQUIRIES","AVERAGE.ACCT.AGE","CREDIT.HISTORY.LENGTH"]]
Y_Train=train_data["loan_default"]
X_Train,X_val,Y_Train,Y_val=train_test_split(X_Train,Y_Train,test_size=0.1)
LogReg=AdaBoostClassifier()
LogReg.fit(X_Train,Y_Train)
y_pred=LogReg.predict(X_val)
print("Validation Accuracy: ",accuracy_score(Y_val,y_pred)*100)
