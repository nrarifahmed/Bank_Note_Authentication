#!/usr/bin/env python
# coding: utf-



####  https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data

import numpy as np
import pandas as pd

df = pd.read_csv('D:/bank_note_auth_streamlit/BankNote_Authentication.csv')
df.head(20)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X
y

#### Train Test SPlit
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

### Implement Random Forest classifier
from sklearn.ensemble import RandomForestClassifier
bank_note_classifier=RandomForestClassifier()
bank_note_classifier.fit(X_train,y_train)

### prediction
y_pred = bank_note_classifier.predict(X_test)

### Check Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
score


###### Create a Pickle File using Serialization
import pickle
pickle_out=open("bank_note_classifier.pkl","wb")
pickle.dump(bank_note_classifier,pickle_out)
pickle_out.close()

import numpy as np
bank_note_classifier.predict([[2,3,4,5]])

import numpy as np
bank_note_classifier.predict([[5,1,4,1]])




