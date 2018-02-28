# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 13:16:07 2018
from sklearn.preprocessing import Imputer
imputer =Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,3])
X[:,]=imputer.transform(X[:,3])
@author: athul
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset=pd.read_csv('hotel_data.csv')

dataset.loc[dataset['Veg/Non-Veg'] =='V', 'Veg/Non-Veg'] = 1
dataset.loc[dataset['Veg/Non-Veg'] =='N', 'Veg/Non-Veg'] = 0

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,4].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer =Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:2])
X[:,1:2]=imputer.transform(X[:,1:2])

imputer2 =Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer2=imputer.fit(X[:,3:4])
X[:,3:4]=imputer2.transform(X[:,3:4])

#Categorizing and encoding
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
X[:,2]=labelencoder_X.fit_transform(X[:,2])

oneHotEncoder=OneHotEncoder(categorical_features=[0])
X=oneHotEncoder.fit_transform(X).toarray()

labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#Test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler

sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
