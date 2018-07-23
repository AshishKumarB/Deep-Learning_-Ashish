# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer, units represent the # neurons in the same
# Dense is used to make a full connected layer
# Drop out helps to avoid over fitting by deactivating some random neurons
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p=  0.1, seed= 123))
# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(0.1, seed= 123))
# Adding the output layer. 1 represents the last neuron
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
# Adam is an efficient way to implement Stochastic Gradient Decent
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Part 4
# Evaluating, Tunning and Improving the ANN
# K fold CV
# Keras does not have a direct CV module. So we use a Keras wrapper around the scikit learn

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score # This is for K fold CV
from sklearn.model_selection import GridSearchCV # This Grid Search + K fold CV
import keras
from keras.models import Sequential
from keras.layers import Dense

# Implement K-fold Cross Validation via Keras
# The KerasClassifier expects a function
# Define the architecture in the function. The function builds the ANN

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6,input_dim= 11, init='uniform', activation ='relu'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Create a new global classifier = classifier_global
classifier = KerasClassifier(build_fn=build_classifier,batch_size= 10, nb_epoch=100)
# return 10 accuracies on test set using 10 fold CV. I implemented 3 fold to cutdown the time
accuracies = cross_val_score(estimator=classifier, X=X_train, y= y_train,cv= 3,n_jobs=-1)
mean_accuracy = accuracies.mean()
var_accuracy = accuracies.std()

# Part 5
# Improving the ANN 
# Parameter tunning using Grid Search
from sklearn.model_selection import GridSearchCV # This Grid Search + K fold CV
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim=6,input_dim= 11, init='uniform', activation ='relu'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
# If you want to tune any argument in the architecture, pass it as an argument and change the variable value
classifier = KerasClassifier(build_fn=build_classifier)
# Batch size is power of 2
params= {'batch_size':[25,32],
         'nb_epoch':[100, 300],
         'optimizer':['adam','rmsprop']}

# Create a Grid Search object from gs clas
# Present accuracy = 83%
grid_search= GridSearchCV(estimator=classifier,
                          param_grid=params,
                          scoring='accuracy',
                          cv= 10)
model_GS_CV= grid_search.fit(X_train,y_train)
best_params=grid_search.best_params_
best_accuracy= grid_search.best_score_

# Implement Dropout Regularization to reduce over fitting