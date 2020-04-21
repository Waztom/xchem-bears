#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:16:39 2020

@author: warren
"""
import rdkit_utils
import pandas as pd
import featuretools as ft
import joblib

# Use the model to select frags
df_test_frags = pd.read_excel("Data/Mpro_cocryst_2020_04_16.xlsx", usecols='D')
df_test_frags.rename(columns={'SMILES': 'CompoundSMILES'}, inplace=True)

df_SMILES_frags = pd.DataFrame(columns=['CompoundSMILES','Pred'])

# Generate rdkit descriptors from SMILES
X_test, properties = rdkit_utils.get_rdkit_properties(df_test_frags)

# Remove SMILES column from X train
df_SMILES_frags['CompoundSMILES'] = X_test['CompoundSMILES']

# Let's try add some feature engineering from feature tools
# Make an entityset and add the entity
es = ft.EntitySet(id = 'chem_features')
es.entity_from_dataframe(entity_id = 'data', dataframe = X_test, 
                         make_index = False, index = 'CompoundSMILES')

# Run deep feature synthesis with transformation primitives
X_test, feature_defs = ft.dfs(entityset = es,
                                      max_depth=1,
                                      target_entity = 'data',
                                      agg_primitives=["mean", "sum", "mode"],
                                      trans_primitives = ['add_numeric', 
                                                          'multiply_numeric',
                                                          'equal'])

# Load saved rf model from classify.py script
rf_model = joblib.load("Models/RF_model.pkl")

# What is the probability score though? First column prob no-binding and
# second column prob of binding
y_test = rf_model.predict_proba(X_test)

# Get SMILES of BRICS_predictions
pred = pd.Series(y_test[:,1])
df_SMILES_frags['Pred'] = pred

# Filter only hits
df_SMILES_frags = df_SMILES_frags[df_SMILES_frags.Pred > 0.90]

# Write to csv
df_SMILES_frags['CompoundSMILES'].to_csv('Hits/Frag_hits.csv', index=False)
