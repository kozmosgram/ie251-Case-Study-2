# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:43:14 2023

@author: Tolga
"""

# pyomo_sens_analysis.py

import numpy as np
import pandas as pd
import re

def read_LP_file(file_path_LP):
    # Read the data into a DataFrame
    df_LP = pd.read_csv(file_path_LP, header=0)
    
    # Find the index of rows where the substrings in components_to_find exists in the df_LP's first column
    components_to_find = ['c_l_', 'c_u_', 'c_e_', 'bound', 'end']
    index_of_mdlLP_constraints=[]
    index_of_mdlLP_variables=[]
    for sub_comp in components_to_find:
        if sub_comp != 'bound' and sub_comp != 'end':
            index_of_mdlLP_constraints.append(df_LP[df_LP.iloc[:,0].str.contains(sub_comp)].index)
        else:
            index_of_mdlLP_variables.append(df_LP[df_LP.iloc[:,0].str.contains(sub_comp)].index)
    
    LP_constraint_names = []
    for index in index_of_mdlLP_constraints:
        for sub_index in index:
            LP_constraint_names.append(df_LP.iloc[sub_index,0][:-1])
    
    df_LP_variables = df_LP.iloc[index_of_mdlLP_variables[0][0]+1:index_of_mdlLP_variables[1][0],0].copy()
    
    # Split the entries in the 'Text' column based on the comma ('<=') delimiter
    df_LP_variables_split = df_LP_variables.str.split('<=')
    
    # Expand the list of split values into separate columns and store the second column in a list
    LP_variable_names = df_LP_variables_split.apply(pd.Series).iloc[:,1].to_list()

    return LP_variable_names, LP_constraint_names  

def read_SA_file(file_path_SA):
    # Specify the keyword to use for splitting
    split_keyword = 'No.'
    
    # Read the file and split data into chunks
    with open(file_path_SA, 'r') as file:
        data = file.read()
    
    # Split the data into chunks using the specified keyword
    chunks = data.split(split_keyword)
    
    # Initialize a list to store DataFrames
    data_frames = []
    
    # Process each chunk
    n=0
    for chunk in chunks:
        if not chunk.strip():
            continue
    
        # Split the chunk into lines
        lines = chunk.strip().split('\n')
    
        # Identify the line containing column information
        header_line = lines[2]
    
        # Use regular expression to find positions of consecutive whitespaces
        column_positions = [match.start() for match in re.finditer(r'\s+', header_line)]
    
        # Initialize a list to store rows
        rows = []
    
        # Process each line in the chunk
        for line in lines:
            # Extract data based on identified column positions
            data = [line[pos_start:pos_end].strip() for pos_start, pos_end in zip(column_positions, column_positions[1:]+[None])]
            # Add the data to the current row
            rows.append(data)
        
        # Convert the list of rows into a DataFrame wrt being varible/constraint-related
        if n>0:
            if 'c_l_x' in rows[3][0] or 'c_u_x' in rows[3][0] or 'c_e_x' in rows[3][0] or 'c_e_ONE_VAR' in rows[3][0]:
                df = pd.DataFrame(rows, columns=['Row name', 'St', 'Activity', ' Slack_Marginal', 'Lower bound_Upper bound', \
                                         'Activity range', 'Obj.Coeff range', 'Obj value at break point variable', 'Limiting variable'])
            elif 'x' in rows[3][0] or 'ONE_VAR_CONSTANT' in rows[3][0]:
                df = pd.DataFrame(rows, columns=['Column name', 'St', 'Activity', ' Obj coef_Marginal', 'Lower bound_Upper bound', \
                                         'Activity range', 'Obj.Coeff range', 'Obj value at break point variable', 'Limiting variable'])
        else:
            df = pd.DataFrame(rows)
            n+=1
                    
    
        # Append the DataFrame to the list
        data_frames.append(df)
        
    #Concatenate dataframes wrt being varible/constraint-related
    dfs_variables = pd.DataFrame(columns=['Column name', 'St', 'Activity', ' Obj coef_Marginal', 'Lower bound_Upper bound', \
                             'Activity range', 'Obj.Coeff range', 'Obj value at break point variable', 'Limiting variable'])
    dfs_constraints = pd.DataFrame(columns=['Row name', 'St', 'Activity', ' Slack_Marginal', 'Lower bound_Upper bound', \
                             'Activity range', 'Obj.Coeff range', 'Obj value at break point variable', 'Limiting variable']) 
    n=0
    for data_frame in data_frames:
        if n>0:
            if n<len(data_frames)-1:
                df = data_frame[3:-5]
                df = df.reset_index(drop=True)
            elif n == len(data_frames)-1:
                df = data_frame[3:-2]
                df = df.reset_index(drop=True)
            if 'Column name' in df.columns:
                dfs_variables = pd.concat([dfs_variables, df])
            elif 'Row name' in df.columns:
                dfs_constraints = pd.concat([dfs_constraints, df])
        n+=1
        
     # Drop rows with empty strings in all columns
    dfs_variables.replace('', pd.NA, inplace=True)
    dfs_variables.dropna(how='all', inplace=True)
    dfs_constraints.replace('', pd.NA, inplace=True)
    dfs_constraints.dropna(how='all', inplace=True)
    
    SA_constraint_names=dfs_constraints[dfs_constraints['Row name'].notna()]['Row name'].to_list()
    SA_variable_names=dfs_variables[dfs_variables['Column name'].notna()]['Column name'].to_list()

    return SA_variable_names, SA_constraint_names, dfs_variables, dfs_constraints


def reorganize_SA_report(file_path_SA, file_path_LP_labels, file_path_LP_nolabels):
    """
    This is a brief explanation of the function "reorganize_SA_report" in the module "pyomo_sens_analysis".

    Inputs:
    - file_path_SA: Specify the path of your sensitivity analysis results file stored in a text file (e.g., 'D:\YourFilePath\YourFileName.txt').
    - file_path_LP_labels: Specify the path of model file stored in a LP file with user-defined labels generated by "io_options={'symbolic_solver_labels': True}" (e.g., 'D:\YourFilePath\YourFileName.lp').
    - file_path_LP_nolabels: Specify the path of model file stored in a LP file without user-defined labels generated by "io_options={'symbolic_solver_labels': False}"  (e.g., 'D:\YourFilePath\YourFileName.lp').
    Output:
    Excel file containing sensitivity analysis results for obj. func. coeff. and constraint RHS in "Variables" and "Constraints" sheets, respectively.
    """
    (SA_variable_names, SA_constraint_names, dfs_variables, dfs_constraints) = read_SA_file(file_path_SA)
    
    (LP_variable_names_labels, LP_constraint_names_labels) =read_LP_file(file_path_LP_labels)
    
    (LP_variable_names_nolabels, LP_constraint_names_nolabels) =read_LP_file(file_path_LP_nolabels)
    
    LP_variable_name_mapping = {}
    j=0
    for i in LP_variable_names_nolabels:
        LP_variable_name_mapping[i]=LP_variable_names_labels[j]
        j+=1    
    
    LP_constraint_name_mapping = {}
    j=0
    for i in LP_constraint_names_nolabels:
        LP_constraint_name_mapping[i]=LP_constraint_names_labels[j]
        j+=1 
    
    dfs_variables_2 = dfs_variables.copy()
    j=0
    for i in np.arange(dfs_variables.shape[0]):
        if j<len(LP_variable_name_mapping)-1 and isinstance(dfs_variables.iloc[i,0], str):
            dfs_variables.iloc[i,0]=LP_variable_name_mapping[' ' + dfs_variables_2.iloc[i,0] + ' ']
            j+=1
    
    dfs_constraints_2 = dfs_constraints.copy()
    j=0
    for i in np.arange(dfs_constraints.shape[0]):
        if j<len(LP_constraint_name_mapping)-1 and isinstance(dfs_constraints.iloc[i,0], str):
            dfs_constraints.iloc[i,0]=LP_constraint_name_mapping[dfs_constraints_2.iloc[i,0]]
            j+=1
    
    # Create an ExcelWriter object
    with pd.ExcelWriter('Sensitivity_Analysis_Report.xlsx') as writer:
        # Write each DataFrame to a different sheet
        dfs_variables.to_excel(writer, sheet_name='Variables', index=False)
        dfs_constraints.to_excel(writer, sheet_name='Constraints', index=False)
    return
