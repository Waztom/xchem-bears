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
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
import joblib
    
# Read csv of John's docking scores 
script_dir = os.path.dirname(os.path.abspath("__file__"))
rel_path = "Data/Docking_03042020.csv"
abs_file_path = os.path.join(script_dir, rel_path)

docked_df = pd.read_csv(abs_file_path, usecols=["SMILES", "Hybrid2"])
docked_df.columns = ["CompoundSMILES", "Hybrid2"] 

# Produce rdkit features from SMILES
df,properties = rdkit_utils.get_rdkit_properties(docked_df)

# Get X, y and training and test data
y = df['Hybrid2']
X = df.drop(columns=['CompoundSMILES', 'Hybrid2'])

# Normalise data
scaler_data = MinMaxScaler(feature_range = (-1, 1))

#Scale data used for model
X = scaler_data.fit_transform(X)

#Save scaler model for running submission
scaler_filename = "Data/scaler_data.save"
joblib.dump(scaler_data, scaler_filename)
    
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.33, 
                                                    shuffle=True)

'''
'Let's create a RF model instance. Seed to see same results. High trees to
try avoid OF 
'''

# n_estimators = [500, 1000, 5000, 10000]
# max_features = [5, 10, 15, 20]
# max_depth = [5, 10, 15, 20]
# min_samples_split = [2, 5, 10, 15, 100]
# min_samples_leaf = [1, 2, 5, 10] 

# hyperF = dict(n_estimators = n_estimators,
#               max_features = max_features,
#               max_depth = max_depth,  
#               min_samples_split = min_samples_split, 
#               min_samples_leaf = min_samples_leaf)

# forest = RandomForestRegressor()
# gridF = GridSearchCV(forest, 
#                      hyperF, 
#                      cv = 3, 
#                      verbose = 1, 
#                      n_jobs = -1)

# bestF = gridF.fit(X_train, y_train)

###########
rf_model   = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=20, max_features=5, max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=10000, n_jobs=None, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)

cv_scores = cross_val_score(rf_model, X_train, y_train, cv=3)

# What's the accuracy of our model?
print("Accuracy: {0:0.2f} [Standard {1}]".format(cv_scores.mean(), 
                                                 rf_model.__class__.__name__))

# Fit to training data
rf_model.fit(X_train,y_train)

# Predictions test
y_pred = rf_model.predict(X_test)

# Get the strength of the RF model's feature importance
for name,model_importance in zip(properties.GetPropertyNames(), rf_model.feature_importances_):
    print("Model importance: {0:.2f} for property: {1}".format(model_importance, name))

# Save the model
rel_path = "Models/RF_reg_model.pkl"
abs_file_path = os.path.join(script_dir, rel_path)
joblib.dump(rf_model, abs_file_path, compress=9)

