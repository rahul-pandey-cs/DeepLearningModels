#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 07:57:50 2018

@author: Rahul
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Reading CSV file and extracting variables that are relevant

csv_file = pd.read_csv("Churn_Modelling.csv")
X = csv_file.iloc[:,3:13].values
Y = csv_file.iloc[:,13].values

chk = csv_file.iloc[:,3:13].values

#Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

oneHotEncoder = OneHotEncoder(categorical_features=[1])
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:,1:]


# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


## Feature Scaling

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Lets make the ANN
# Importing the keraas libraries and packages

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))

# Adding the output hidden layer
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

#Compiling the ANN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train,Y_train, batch_size = 10, epochs=100)

# Making predictions
# Predicting Test set results

y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Making the confusion metrics

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

X_test[0:1,:] #Reading first row of the array

test_pred = classifier.predict(X_test[0:1,:])
test_pred = (test_pred>0.5)

#Predicting for one input
#Geography: France, Credit Score: 600, Gender: Male, Age: 40 years old, Tenure: 3 years, Balance: $60000, Number of Products: 2
#Does this customer have a credit card ? Yes, Is this customer an Active Member: Yes, Estimated Salary: $50000

#CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Exited

input_val = np.array([[600,'France','Male',40,3,60000.0,2,1,1,50000.0]], dtype=object)

#Encoding categorical data

labelencoder_X_1_inp = LabelEncoder()
input_val[:, 1] = labelencoder_X_1_inp.fit_transform(input_val[:, 1])

labelencoder_X_2_inp = LabelEncoder()
input_val[:, 2] = labelencoder_X_2_inp.fit_transform(input_val[:, 2])

oneHotEncoder_inp = OneHotEncoder(categorical_features=[1])
input_val = oneHotEncoder_inp.fit_transform(input_val).toarray()
input_val = input_val[:,1:]


## Feature Scaling

inp_scaler = StandardScaler()
input_val = inp_scaler.fit_transform(input_val)

#Predicting outcome for single input
input_pred = classifier.predict(np.array([[0.0,0,600,1,40,3,60000.0,2,1,1,50000.0]]))
input_pred = (test_pred>0.5)


## Evaluating, Improving and Tuning the ANN Model performance
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size =10, epochs=100)
accuracies = cross_val_score(estimator=classifier,X=X_train,y=Y_train, cv =10, n_jobs=-1 )

mean = accuracies.mean()
variance = accuracies.std()
    







