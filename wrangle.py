import pandas as pd
import numpy as np
import os
import env
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# fips = county
#6039 - LA
#
#

# **************** FUNCTION TO GET SQL CONNECTION *****************
# Create helper function to get the necessary connection url.

def get_zillow():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        print('File exists pulling from system.')
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        print('No file exists, extracting from MySQL.')
        df = new_zillow_data()
        
        # Cache data
        df.to_csv('zillow.csv')

def summarize(df):
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe. For
    each column, it returns the column name, the number of
    unique values in the column, the unique values themselves,
    the number of null values in the column, and the data type of the column.
    The resulting dataframe is sorted by the 'Number of Unique Values' column in ascending order.

    returns:
        pandas dataframe
    """
    data = []
    # Loop through each column in the dataframe
    for column in df.columns:
        # Append the column name, number of unique values, unique values, number of null values, and data type to the data list
        data.append(
            [
                column,
                df[column].nunique(),
                df[column].unique(),
                df[column].isna().sum(),
                df[column].dtype
            ]
        )

        check_columns = pd.DataFrame(
        data,
        columns=[
            "Column Name",
            "Number of Unique Values",
            "Unique Values",
            "Number of Null Values",
            "dtype"],
    ).sort_values(by="Number of Unique Values")
   
    # Create a pandas dataframe from the data list, with column names 'Column Name', 'Number of Unique Values', 'Unique Values', 'Number of Null Values', and 'dtype'
    # Sort the resulting dataframe by the 'Number of Unique Values' column in ascending order
    return check_columns

def summarize_nums(df):
    '''
    '''
    describe = df.describe()
    info = df.info()
    return describe, info

def zillow(database='zillow',user=env.user, password=env.password, host=env.host):
    '''
    Pulls data from mySql server and drops duplicate columns and values (keeps 1 of needed)
    encodes all categorical data and drops columns that are unnecessary as a by product of 
    new dummy columns.
    '''
    if os.path.isfile('zillow.csv'):
    
    # If csv file exists read in data from csv file.
        zillow = pd.read_csv('zillow.csv', index_col=0)
        
    else:
         # pulling data from mysql
        query = 'SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt,\
                yearbuilt, taxamount, fips, propertylandusetypeid\
                FROM properties_2017\
                WHERE propertylandusetypeid = 261'
        connection = f'mysql+pymysql://{user}:{password}@{host}/{database}'
        zillow = pd.read_sql(query, connection)

        # removing duplicate columns
        zillow = zillow.loc[:,~zillow.columns.duplicated()].copy()
        # encoding categorical type data
        
        # normalizing numerical data
        # EXAMPLE: telco['total_charges'] = telco['total_charges'].str.replace(' ','0').astype('float')
        # additional columns for visualization
        
        # engineered feature of additional services
        
        # dropping extra columns after encoding

        # restoring 'drop_first' column for contract_type as it is desired to specify just this value type (without deducting)

        # lowering all column names
        zillow.columns = map(str.lower,zillow.columns)
        zillow.columns = zillow.columns.str.replace(' ','_')

        # cache data
        zillow.to_csv('zillow.csv')
    return zillow
    
### To get single family homes not in a query ###
    #single_family = zillow[zillow['propertylandusetypeid'] == 261]
    #single_family

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