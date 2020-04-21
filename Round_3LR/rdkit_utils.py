#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:21:02 2020

@author: Warren Thompson (Waztom)

BRICS and rdkit feature functions
"""
from rdkit.Chem import MolFromSmiles, MolToSmiles, rdMolDescriptors, SanitizeMol, CanonSmiles
from rdkit.Chem.BRICS import BRICSDecompose
import csv

def standardise_SMILES(SMILES):
    mols = get_mol_objects(SMILES)
    return get_SMILES_objects(mols)

def get_SMILES_objects(mols):
    if type(mols) == list:
        SMILES = [MolToSmiles(mol) for mol in mols]
        return [CanonSmiles(SMILES) for SMILES in SMILES]
    if str(type(mols)) == "<class 'rdkit.Chem.rdchem.Mol'>":
        SMILES = MolToSmiles(mols)
        return CanonSmiles(SMILES)
    
def get_mol_objects(SMILES):
    if type(SMILES) == list:
        mols = [MolFromSmiles(SMILES) for SMILES in SMILES]
        return mols
    if type(SMILES) == str:
        mol = MolFromSmiles(SMILES)
        return mol

def get_rdkit_properties(df):
    properties = rdMolDescriptors.Properties()
    mol_list = [MolFromSmiles(SMILES) for SMILES in df.CompoundSMILES]
       
    descriptor_values = []
    descriptor_names = [name for name in properties.GetPropertyNames()]

    # Add all of the avilable rdkit descriptors
    for mol in mol_list:
        descriptor_temp_list = []
        for name,value in zip(properties.GetPropertyNames(), properties.ComputeProperties(mol)):
            descriptor_temp_list.append(value)
        descriptor_values.append(descriptor_temp_list)
    
    i = 0    
    for name in descriptor_names:
        list_append = [value[i] for value in descriptor_values]
        df[name] = list_append
        i += 1
    
    return df,properties  

def get_BRICS(mol_list, min_frag_size, keep_nodes):
    allfrags=set()
    for mol in mol_list:        
        frags = BRICSDecompose(mol,
                               keepNonLeafNodes=keep_nodes,
                               minFragmentSize=min_frag_size)
        allfrags.update(frags)
    allfrags = get_mol_objects(sorted(allfrags))
    return allfrags

def get_lipinksi_test(mol, rule_test):
    mol.UpdatePropertyCache(strict=False)  
    MW = rdMolDescriptors.CalcExactMolWt(mol)
    
    # Calculate mol features. NB CalcCrippenDescriptors returns tuple logP & mr_values
    feature_values = [rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
                      rdMolDescriptors.CalcNumLipinskiHBD(mol),
                      rdMolDescriptors.CalcNumLipinskiHBA(mol)]
    test_rule = all(value <= rule_test for value in feature_values)
    if MW < 500 and MW > 300 and test_rule == True:
        return True
    else:
        return False

def get_BRICS_builds(BRICS_func, rule_test, block_size=1000):
    # Will do this in blocks to avoid running out of memory
    block = []   
    for mol in BRICS_func:   
        if get_lipinksi_test(mol,rule_test) == True:
            SanitizeMol(mol)
            block.append(mol)
        if len(block) == block_size:
            yield block
            block = [] 
            
    # Yield the last block
    if block:
        yield block
        
def get_filtered_frags(frag_list, pattern_list):
    frag_mol = get_mol_objects(frag_list)
    for pattern in pattern_list:
        patt = get_mol_objects(pattern)
        for mol in frag_mol:
            if mol.HasSubstructMatch(patt):
                frag_mol.remove(mol)
    return frag_mol

def write_BRICS_csv(BRICS_builds_gen, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(['CompoundSMILES'])
        for mol_list in BRICS_builds_gen:
            for mol in mol_list:
                mol.UpdatePropertyCache(strict=False)
            prods = [[MolToSmiles(mol)] for mol in mol_list]
            writer.writerows(prods)



