#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 11:21:02 2020

@author: Warren Thompson (Waztom)

BRICS and rdkit feature functions
"""
from rdkit.Chem import MolFromSmiles, MolToSmiles, rdMolDescriptors 
from rdkit.Chem.BRICS import BRICSDecompose, BRICSBuild
import csv

def get_mol_objects(SMILES):
    if type(SMILES) == list:
        return [MolFromSmiles(SMILES) for SMILES in SMILES]
    if type(SMILES) == str:
        return MolFromSmiles(SMILES)

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
    allfrags = [MolFromSmiles(SMILES) for SMILES in sorted(allfrags)]
    return allfrags

def get_lipinksi_test(mol):
    mol.UpdatePropertyCache(strict=False)  
    MW = rdMolDescriptors.CalcExactMolWt(mol)
    
    # Calculate mol features. NB CalcCrippenDescriptors returns tuple logP & mr_values
    feature_values = [rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
                      rdMolDescriptors.CalcNumLipinskiHBD(mol),
                      rdMolDescriptors.CalcNumLipinskiHBA(mol)]
    test_5 = all(value <= 5 for value in feature_values)
    if MW < 500 and MW > 300 and test_5 == True:
        return True
    else:
        return False

def get_BRICS_builds(BRICS_func, block_size=1000):
    # Will do this in blocks to avoid running out of memory
    block = []    
    for mol in BRICS_func:   
        if get_lipinksi_test(mol) == True:
            block.append(mol)
        if len(block) == block_size:
            yield block
            block = [] 
            
    # Yield the last block
    if block:
        yield block
        
def get_filtered_frags(frag_list, pattern_list):
    frag_mol = [MolFromSmiles(SMILES) for SMILES in frag_list]
    for pattern in pattern_list:
        patt = MolFromSmiles(pattern)
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



