#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:48:26 2020

@author: Warren Thompson (Waztom)

Enumeration of fragments
"""
import pandas as pd
import rdkit_utils
import random
import joblib
from rdkit.Chem.BRICS import BRICSBuild

hit_df = pd.read_csv("Data/hits.csv")

#Remove hits not binding to active site to create fragemnts from
hit_df = hit_df[hit_df.Site_No == 1]

# Creat mol objects from SMILES
mol_list = rdkit_utils.get_mol_objects(list(hit_df.CompoundSMILES))

# Let's direct some of the builds by adding frags we think are important
frags_to_add = ["N#Cc1ccc([16*])c([16*])c1"]

for frag in frags_to_add:
    mol_list.append(rdkit_utils.get_mol_objects(frag))

# Break fragments into synthetic biulding blocks using BRICS algo
allfrags = rdkit_utils.get_BRICS(mol_list, 
                                 min_frag_size=8)
        
# Build fragments from BRICS
random.seed(127)
random.seed(0xf00d)
BRICS_func = BRICSBuild(allfrags)
       
# Get BRICS builds and allocate them to list chunks in generator 
# to help memory
all_BRICS_builds = rdkit_utils.get_BRICS_builds(BRICS_func)

# Convert to SMILES and write to csv for having a look at later
rdkit_utils.write_BRICS_csv(all_BRICS_builds,
                            filename="Data/BRICS_hits.csv")

# Use the model to filter the BRICS compounds
df_test_BRICS = pd.read_csv("Data/BRICS_hits.csv", header=0)
df_SMILES_BRICS = pd.DataFrame(columns=['CompoundSMILES','Pred'])

# Generate rdkit descriptors from SMILES
X_test_BRICS_pred, properties = rdkit_utils.get_rdkit_properties(df_test_BRICS)

# Filter using Lipinksi filter/NB >300 and <500 MW applied in BRICS.py
X_test_BRICS_pred = X_test_BRICS_pred[X_test_BRICS_pred.NumRotatableBonds <= 5]

# Remove SMILES column from X train
df_SMILES_BRICS['CompoundSMILES'] = X_test_BRICS_pred['CompoundSMILES']
X_test_BRICS_pred = X_test_BRICS_pred.drop(columns=['CompoundSMILES'])

# Load saved rf model from classify.py script
rf_model = joblib.load("Models/RF_model.pkl")

# Test model 
y_test_BRICS_pred = rf_model.predict(X_test_BRICS_pred)

# Get SMILES of BRICS_predictions
pred = pd.Series(y_test_BRICS_pred)
df_SMILES_BRICS['Pred'] = pred

# Filter only hits
df_SMILES_BRICS = df_SMILES_BRICS[df_SMILES_BRICS.Pred == 1]

# Write to csv
df_SMILES_BRICS['CompoundSMILES'].to_csv('Hits_from_frags/BRICS_hits.csv', index=False)



