#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 14:08:47 2020

@author: warren
Possible NN
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Let's try some NN models)

# The dataset knows the number of features, e.g. 2
n_features = X.shape[1]

# Define model
# create model
def baseline_model():
    model = Sequential()
    model.add(Dense(200, input_dim=n_features, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
       
model = baseline_model()
 
# Fit model
estimator = KerasRegressor(build_fn=baseline_model, 
                            nb_epoch=50, 
                            batch_size=1, 
                            verbose=0)

# Cross validation
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

model.fit(X_train, y_train)

# Demonstrate prediction
y_pred_NN = model.predict(X_test, verbose=1)

# Convert to pandas series for calcs
y_pred_NN = pd.DataFrame(y_pred_NN)

# R2 score
err2 = sum((y_test - y_pred_NN) ** 2)
y2_tally = sum(y_test ** 2)

r2 = 1 - err2 / y2_tally

print(r2)

model.save('Models/NN_reg.h5')