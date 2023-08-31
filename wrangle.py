import pandas as pd
import numpy as np
import os
import env
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer



# **************** FUNCTION TO GET SQL CONNECTION *****************
# Create helper function to get the necessary connection url.

def get_connection_url(db, username=env.username, host=env.host, password=env.password):
    """
    This function will:
    - take username, pswd, host credentials from imported env module
    - output a formatted connection_url to access mySQL db
    """
    return f'mysql+pymysql://{username}:{password}@{host}/{db}'


def get_zillow():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_data()
        
        # Cache data
        df.to_csv('zillow.csv')
    
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

    return df