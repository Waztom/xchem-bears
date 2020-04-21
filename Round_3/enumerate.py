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

hit_df = pd.read_csv("Data/First_round_hits_less_12_top_40_310320_plus_covalents.csv")

#Remove hits not binding to active site to create fragemnts from
hit_df = hit_df[hit_df.Site_No == 1]

# Creat mol objects from SMILES
mol_list = rdkit_utils.get_mol_objects(list(hit_df.CompoundSMILES))

# Break fragments into synthetic biulding blocks using BRICS algo
allfrags = rdkit_utils.get_BRICS(mol_list, 
                                 min_frag_size=10,
                                 keep_nodes=False)

# Build fragments from BRICS
random.seed(127)
random.seed(0xf00d)
BRICS_func = BRICSBuild(allfrags)
       
# Get BRICS builds and allocate them to list chunks in generator 
# to help memory
all_BRICS_builds = rdkit_utils.get_BRICS_builds(BRICS_func, rule_test=5)

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
rf_model = joblib.load("Models/RF_reg_model.pkl")

# Test model 
y_test_BRICS_pred = rf_model.predict(X_test_BRICS_pred)

# Get SMILES of BRICS_predictions
pred = pd.Series(y_test_BRICS_pred)
df_SMILES_BRICS['Pred'] = pred

# Filter only hits
df_SMILES_BRICS = df_SMILES_BRICS[df_SMILES_BRICS.Pred < 0.9]

# Get sub summary (Need to find out way to do automatically????)
submission_df = pd.read_excel('Data/covid_submissions_03_31_2020.xlsx')
submision_df =[['SMILES']]
submission_df = submission_df.rename(columns={"SMILES": "CompoundSMILES"})
    
# Filter out compounds by merge
df_SMILES_BRICS = (submission_df.merge(df_SMILES_BRICS, 
                                       on = 'CompoundSMILES', 
                                       how='right', 
                                       indicator=True).query('_merge == "right_only"').drop('_merge', 1))

# Write to csv
df_SMILES_BRICS['CompoundSMILES'].to_csv('Hits_from_frags/BRICS_hits.csv', index=False)

###### STUFF to add maybe
# Filter out hits that have already been submitted - first get most up to date 
# sub file
# date = datetime.today().strftime('%Y-%m-%d')

# # Get the XChem fragment summary
# rel_path = "Data/Submission_summary_{}.xlsx".format(date)
# script_dir = os.path.dirname(os.path.abspath("__file__"))
# abs_file_path = os.path.join(script_dir, rel_path)

# try:
#     submission_df = pd.read_excel(abs_file_path)
# except:
#     dls = 'https://docs.google.com/spreadsheets/d/1zELgd-kDEkIjRqc_jdKm5EzDQmRrrYAbErghTPkcA5c/edit#gid=0'   

#     submission_df = pd.read_excel(dls)
#     submision_df =[['SMILES']]
#     submission_df = submission_df.rename(columns={"SMILES": "CompoundSMILES"})
#     submision_df.to_csv(abs_file_path, 
#                 index=False)
   

