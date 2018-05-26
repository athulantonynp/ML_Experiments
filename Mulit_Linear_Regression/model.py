import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Reading data set and converting independent columns to 
#X and dependent column to y

data=pd.read_csv('50_Startups.csv')
X=data.iloc[:,:-1].values
y=data.iloc[:,4].values

#Transforming categorical datas to encoded data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Removing unwanted category.Dummy variable trap removal.
X = X[:, 1:]

#Forming test and train set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Creating regression.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting.
y_pred = regressor.predict(X_test)
