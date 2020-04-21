# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:58:10 2020

@author: Warren Thompson (Waztom)

Regression model from John Chodera's docking
scores for enumerating fragments for COVID Moonshot
Round 2 subs
"""
import rdkit_utils
import os
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
import joblib
import featuretools as ft
import matplotlib.pyplot as plt
from math import sqrt
    
# Read csv of John's docking scores 
file_path = "Data/covid_submissions_all_info-2020-04-06-docked-justscore.csv"

docked_df = pd.read_csv(file_path, usecols=["SMILES", "Hybrid2"])
docked_df.columns = ["CompoundSMILES", "Hybrid2"] 

# Get rid of duplicate values
docked_df = docked_df.drop_duplicates(subset='CompoundSMILES', keep="first")
docked_df = docked_df.reset_index(drop=True)

# Produce rdkit features from SMILES
df,properties = rdkit_utils.get_rdkit_properties(docked_df)

# Get X, y and training and test data
y = df['Hybrid2']
X = df.drop(columns=['Hybrid2'])

# Let's try add some feature engineering from feature tools
# Make an entityset and add the entity
es = ft.EntitySet(id = 'chem_features')
es.entity_from_dataframe(entity_id = 'data', dataframe = X, 
                         make_index = False, index = 'CompoundSMILES')

# Run deep feature synthesis with transformation primitives
X, feature_defs = ft.dfs(entityset = es,
                                      max_depth=1,
                                      target_entity = 'data',
                                      agg_primitives=["mean", "sum", "mode"],
                                      trans_primitives = ['add_numeric', 
                                                          'multiply_numeric',
                                                          'cum_count',
                                                          'cum_mean',
                                                          'cum_sum',
                                                          'equal'])


'''
'Let's create a RF model instance. Seed to see same results. High trees to
try avoid OF 
'''
rf_model   = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=15, max_features=20, max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=5,
                      min_samples_split=5, min_weight_fraction_leaf=0.0,
                      n_estimators=3000, n_jobs=-1, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)

# Create and fit selector
selector = SelectFromModel(rf_model)
selector.fit(X, y)

# Get columns to keep and create new dataframe with those only
cols = selector.get_support(indices=True)
X = X.iloc[:,cols]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.33, 
                                                    shuffle=True)

cv_scores = cross_val_score(rf_model, X_train, y_train, cv=3)

# What's the accuracy of our model?
print("Accuracy: {0:0.2f} [Standard {1}]".format(cv_scores.mean(), 
                                                 rf_model.__class__.__name__))


# Fit to training data
rf_model.fit(X_train,y_train)

# Predictions test
y_pred = rf_model.predict(X_test)

# RMSE
rms = sqrt(mean_squared_error(y_test, y_pred))
print("The RMSE is: {}".format(rms))

# Plot the predictions and test data
# First need to sort out y_test index
y_test.reset_index(drop=True, inplace=True)

# Visualise
plt.scatter(y_test, y_pred)
plt.xlabel("Test score")
plt.ylabel("Pred score")
plt.title("Comparison of random selection test data and predicted Hybrid$^2$ scores \n RMSE = {:.2f}".format(rms))
plt.show()

# Save the model
file_path = "Models/RF_reg_model.pkl"
joblib.dump(rf_model, file_path, compress=9)

# How good is the model at predicting stuff outside the dataset?
# Compare to docked scores for x-ray data
xray_df = pd.read_csv("Data/all-screened-fragments-docked.csv",
                      usecols=['SMILES', 'Hybrid2'])
xray_df = xray_df.rename(columns={"SMILES": "CompoundSMILES"}) 

# Get rid of duplicate values
xray_df = xray_df.drop_duplicates(subset='CompoundSMILES', 
                                              keep="first")

xray_df = xray_df.reset_index(drop=True)

# Get y actual
y_test_xray = xray_df.drop(columns=['CompoundSMILES'])

# Prep X_test_xray
X_test_xray = xray_df.drop(columns=['Hybrid2'])

# Produce rdkit features from SMILES
X_test_xray,properties = rdkit_utils.get_rdkit_properties(X_test_xray)

# Let's try add some feature engineering from feature tools
# Make an entityset and add the entity
es = ft.EntitySet(id = 'chem_features')
es.entity_from_dataframe(entity_id = 'data', dataframe = X_test_xray, 
                         make_index = False, index = 'CompoundSMILES')

# Run deep feature synthesis with transformation primitives
X_test_xray, feature_defs = ft.dfs(entityset = es,
                                      max_depth=1,
                                      target_entity = 'data',
                                      agg_primitives=["mean", "sum", "mode"],
                                      trans_primitives = ['add_numeric', 
                                                          'multiply_numeric',
                                                          'cum_count',
                                                          'cum_mean',
                                                          'cum_sum',
                                                          'equal'])
                               
# Select top features
X_test_xray = X_test_xray.iloc[:,cols]

# Predict Hybrid2
y_pred_xray = rf_model.predict(X_test_xray)
y_pred_xray = pd.DataFrame(data=y_pred_xray.flatten(), columns=['Hydrid2'])

# RMSE
rms = sqrt(mean_squared_error(y_test_xray, y_pred_xray))
print("The RMSE is: {}".format(rms))

# Visualise
plt.plot(y_test_xray, "b", label="Original score")
plt.plot(y_pred_xray, "r", label="Pred score")
plt.ylabel("Hybrid$^2$ score")
plt.xlabel("Number of measurements")
plt.title("Comparison screening fragments docked scores with predicted scores \n RMSE = {:.2f}".format(rms))
plt.legend()
plt.show()