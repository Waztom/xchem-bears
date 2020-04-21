#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:48:26 2020

@author: Warren Thompson (Waztom)

Get data from SQL DB
"""
import SQL_utils

'''
File path to soakDB. I used a little bit of a dated version. This can
easily be set to the soakDB path in the XChem folder on the server
'''
db_file = 'Data/soakDBDataFile.sqlite'

# Create SQL connection and dataframe
conn = SQL_utils.create_connection(db_file)
df = SQL_utils.select_data(conn)

# Call the filter function and write to different files
library_list = ["All","CovLibrary",
                "DMSO","DO_Cmpd",
                "DSIpoised_DMSO",
                "Probing fragment all"]
for library_name in library_list:
    SQL_utils.filter_df(library_name, df)

