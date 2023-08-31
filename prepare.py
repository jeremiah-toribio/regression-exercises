import pandas as pd
import numpy as np
import env
import acquire
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



def prep_zillow():
    '''
    WARNING: This will be in conjunction with the acquire.py file and without such will not function.
    For operation on acquire.py, please see repository.

    Pulls data from mySql server and drops duplicate columns and values (keeps 1 of needed)
    encodes all categorical data and drops columns that are unnecessary as a by product of 
    new dummy columns.
    '''
    # pulling data from mysql using get_telco_data
    telco = acquire.get_telco_data('telco_churn')
    # removing duplicate columns
    telco = telco.loc[:,~telco.columns.duplicated()].copy()
    # encoding categorical type data
    
    # normalizing numerical data
    
    # additional column for visualization
    
    # engineered feature of additional services
    
    # dropping extra columns after encoding

    # restoring 'drop_first' column for contract_type as it is desired to specify just this value type (without deducting)
   
    # lowering all column names
    zillow.columns = map(str.lower,zillow.columns)
    zillow.columns = zillow.columns.str.replace(' ','_')

    return zillow
    

def splitter(df,target='churn'):
    '''
    Returns
    Train, Validate, Test from SKLearn
    Sizes are 60% Train, 20% Validate, 20% Test
    '''
    train, test = train_test_split(df, test_size=.2, random_state=4343, stratify=df[target])

    train, validate = train_test_split(train, test_size=.2, random_state=4343, stratify=train[target])
    print(f'Dataframe: {df.shape}', '100%')
    print()
    print(f'Train: {train.shape}', '| ~60%')
    print(f'Validate: {validate.shape}', '| ~20%')
    print(f'Test: {test.shape}','| ~20%')
    return train, validate, test