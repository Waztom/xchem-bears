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
from sklearn.model_selection import cross_val_score, train_test_split
import joblib
import featuretools as ft
    
# Read csv of John's docking scores 
script_dir = os.path.dirname(os.path.abspath("__file__"))
rel_path = "Data/Docking_03042020.csv"
abs_file_path = os.path.join(script_dir, rel_path)

docked_df = pd.read_csv(abs_file_path, usecols=["SMILES", "Hybrid2"])
docked_df.columns = ["CompoundSMILES", "Hybrid2"] 

# Need to standardise SMILES 
docked_df.CompoundSMILES = rdkit_utils.standardise_SMILES(list(docked_df.CompoundSMILES))

# Get rid of duplicate values
docked_df = docked_df.drop_duplicates(subset='CompoundSMILES', keep="first")

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

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.33, 
                                                    shuffle=True)

'''
'Let's create a RF model instance. Seed to see same results. High trees to
try avoid OF 
'''
rf_model   = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=5, max_features=200, max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=3000, n_jobs=-1, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)

cv_scores = cross_val_score(rf_model, X_train, y_train, cv=3)

# What's the accuracy of our model?
print("Accuracy: {0:0.2f} [Standard {1}]".format(cv_scores.mean(), 
                                                 rf_model.__class__.__name__))

# Fit to training data
rf_model.fit(X_train,y_train)

# Predictions test
y_pred = rf_model.predict(X_test)

# Save the model
rel_path = "Models/RF_reg_model.pkl"
abs_file_path = os.path.join(script_dir, rel_path)
joblib.dump(rf_model, abs_file_path, compress=9)

