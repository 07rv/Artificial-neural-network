# Artificial Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,13 ].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import  ColumnTransformer

labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])
labelencoder_X_sex = LabelEncoder()
X[:, 2] = labelencoder_X_sex.fit_transform(X[:, 2])

oneHotEncoder_X = OneHotEncoder()
ct = ColumnTransformer(transformers=[('encode',oneHotEncoder_X,[1])],remainder='passthrough')
X = ct.fit_transform(X)

# Remove dummy variable trap
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 2. Making the Artificial Nuural Network (ANN)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input and first hidden layer with dropout
classifier.add(Dense(units=6,activation='relu',input_shape=(11,)))
classifier.add(Dropout(rate=0.1))

# Adding the second hidden layer
classifier.add(Dense(units=6,activation='relu'))
classifier.add(Dropout(rate=0.1))

# Adding the out layer
classifier.add(Dense(units=1,activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', 
                   loss='binary_crossentropy', 
                   metrics=['accuracy'])

# 3. Making predictions and evaluating the model.

# Fitting the ANN to the training set
    classifier.fit(X_train,y_train,batch_size=10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred = (y_pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# =>> Predicting a single new observation
"""
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
"""
single_data = np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])
single_data = sc.transform(single_data)
single_predict = classifier.predict(single_data)
single_predict = (single_predict>0.5)


# 4. Evaluating, Improving and Turning the ANN

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6,activation='relu',input_shape=(11,)))
    classifier.add(Dense(units=6,activation='relu'))
    classifier.add(Dense(units=1,activation='sigmoid'))
    classifier.compile(optimizer='adam', 
                   loss='binary_crossentropy', 
                   metrics=['accuracy'])
    
    return classifier


classifier = KerasClassifier(build_fn=build_classifier,batch_size=10,
                             epochs=100)

accuracies = cross_val_score(estimator = classifier,
                             X = X_train, y = y_train, cv =10)

mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
"""
If model overfits or underfits
"""

# Tunning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6,activation='relu',input_shape=(11,)))
    classifier.add(Dense(units=6,activation='relu'))
    classifier.add(Dense(units=1,activation='sigmoid'))
    classifier.compile(optimizer=optimizer, 
                   loss='binary_crossentropy', 
                   metrics=['accuracy'])
    
    return classifier
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [25, 32],
               'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search =  GridSearchCV(estimator = classifier,
                            param_grid = parameters,
                            scoring='accuracy',
                            cv=10)


grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_ 
best_accuracy  =  grid_search.best_score_










 
