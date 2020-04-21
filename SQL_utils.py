# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:57:41 2020

@author: bvh64415

Access XChem SQLite DB
"""

import sqlite3
import pandas as pd

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except:
        print('Can not make connection')
 
    return conn

def select_data(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return: Pandas df
    """
    SQL_query = '''SELECT 
                       LibraryName,
                       CrystalName,
                       CompoundCode,
                       CompoundSMILES,
                       MountingResult,
                       DataCollectionOutcome,
                       RefinementOutcome,
                       DataProcessingResolutionHigh,
                       DimpleRcryst,
                       DimpleRfree,
                       RefinementRcryst,
                       RefinementRfree
                              
                      FROM 
                      mainTable'''
                      
    df = pd.read_sql_query(SQL_query, conn)
    
    return df

def filter_df(library_name, df):
    '''
    Function call to filter by librart name
    Input: String of library name and the entire df generated from the
    select_data function 
    Output: Filtered df of library selected 
    '''
    
    if library_name == 'All':
        library_name = 'All_libraries'
        # Remove rows where LibraryName == nothing entered
        df = df[df.LibraryName != ''] 
    else:
        # Filter df to include on library
        df = df[df.LibraryName == library_name]
    
    # Let's get rid of the DMSO data
    df = df[df.CompoundCode != 'DMSO']
    
    # Let's drop the Na values from the mounting results
    df = df[df['MountingResult'].notna()]
    
    # Filter into dfs that succeeded\failed mounting
    mounting_fail_filter = "FAIL|Fail"
    mounting_success_filter = "OK|Mounted_Clear|Mounted_Crystalline"
    
    df_failed_mounting =  df[df['MountingResult'].str.contains(mounting_fail_filter)]
    df_success_mounting =  df[df['MountingResult'].str.contains(mounting_success_filter)]
    
    # Let's filter dfs into data success\failed
    data_fail_filter = "None|Failed"
    
    df_failed_data = df_success_mounting[df_success_mounting['DataCollectionOutcome'].str.contains(data_fail_filter)]
    df_success_data= df_success_mounting[df_success_mounting.DataCollectionOutcome == 'success']
       
    # How many true negatives?
    true_neg_filter = "1 - Analysis Pending|2 - PANDDA model|3 - In Refinement"

    df_true_negatives= df_success_data[df_success_data['RefinementOutcome'].str.contains(true_neg_filter)]
    
    # How many hits/true positives?
    true_pos_filter = "5 - Deposition ready|6 - Deposited|4 - CompChem ready"
    
    df_true_positives = df_success_data[df_success_data['RefinementOutcome'].str.contains(true_pos_filter)]
    
    # Sumamry of data
    df_summary = pd.DataFrame(columns = ['No mounting attempts',
                                         'No mounts success',
                                         'No mounts failed', 
                                         'No data success',
                                         'No data failed',
                                         'No hits',
                                         'No misses',
                                         'Mounting success (%)',
                                         'Data success (%)',
                                         'Hit success (%)',
                                          'Mean resolution',
                                          'STD-dev resolution',
                                          'Mean Dimple Rcryst',
                                          'STD-dev Dimple Rcryst',
                                          'Mean Dimple Rfree',
                                          'STD-deb Dimple Rfree'
                                         ])
    
    no_mounting_attempts = len(df) 
    no_mounts_success = len(df_success_mounting) 
    no_mounts_failed = len(df_failed_mounting) 
    no_data_success = len(df_success_data)
    no_data_failed = len(df_failed_data)
    no_hits = len(df_true_positives)
    no_misses = len(df_true_negatives)
    mounting_success = (no_mounts_success / no_mounting_attempts) * 100
    data_success = (no_data_success / no_mounting_attempts) * 100
    hit_success = (no_hits / no_mounting_attempts) * 100
    
    mean_resolution = (df_success_data['DataProcessingResolutionHigh'].astype(float).mean())
    std_resolution = (df_success_data['DataProcessingResolutionHigh'].astype(float).std())
    mean_dimple_rcryst = (df_success_data['DimpleRcryst'].astype(float).mean())
    std_dimple_rcryst = (df_success_data['DimpleRcryst'].astype(float).std())
    mean_dimple_rfree = (df_success_data['DimpleRfree'].astype(float).mean())
    std_dimple_rfree = (df_success_data['DimpleRfree'].astype(float).std())
    
    df_summary.loc[0] = [no_mounting_attempts,
                         no_mounts_success,
                         no_mounts_failed,
                         no_data_success,
                         no_data_failed,
                         no_hits,
                         no_misses,
                         mounting_success,
                         data_success,
                         hit_success,
                         mean_resolution,
                         std_resolution,
                         mean_dimple_rcryst,
                         std_dimple_rcryst,
                         mean_dimple_rfree,
                         std_dimple_rfree
                         ]
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    filepath = 'Data/{}_filtered.xlsx'.format(library_name)   
    
    writer = pd.ExcelWriter(filepath, engine='xlsxwriter')
    
    # Write each dataframe to a different worksheet.
    df_summary.to_excel(writer, sheet_name='Summary',index=False)
    df.to_excel(writer, sheet_name='All_data',index=False)
    
    df_success_mounting.to_excel(writer, sheet_name='Successful mounts',index=False)
    df_failed_mounting.to_excel(writer, sheet_name='Failed mounts',index=False)
    
    df_success_data.to_excel(writer, sheet_name='Successful data',index=False)
    df_failed_data.to_excel(writer, sheet_name='Failed data',index=False)

    df_true_positives.to_excel(writer, sheet_name='Hits',index=False)
    df_true_negatives.to_excel(writer, sheet_name='Misses',index=False)
    
    # Close the Pandas Excel writer and write/save the Excel file.
    writer.save()  
   


