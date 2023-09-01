# Imports ==--==--==
import env
import os
# Ignore Warning
import warnings
warnings.filterwarnings("ignore")
# Array and Dataframes
import numpy as np
import pandas as pd
# Imputer
from sklearn.impute import SimpleImputer
# Evaluation: Visualization
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# Evaluation: Statistical Analysis
from scipy import stats
# Modeling: Scaling
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
# Modeling
from sklearn.model_selection import GridSearchCV
# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# fips = county
#6037 - LA
#6059 - orange
#6111 - Ventura

# **************** FUNCTION TO GET SQL CONNECTION *****************
# Create helper function to get the necessary connection url.

def zillow(database='zillow',user=env.user, password=env.password, host=env.host):
    '''
    Pulls data from mySql server and drops duplicate columns and values (keeps 1 of needed)
    encodes all categorical data and drops columns that are unnecessary as a by product of 
    new dummy columns.
    '''
    if os.path.isfile('zillow.csv'):
        # If csv file exists read in data from csv file.
        print('File exists, pulling from system.')
        zillow = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        # pulling data from mysql
        print('No file exists, extracting from MySQL.')

        query = 'SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt,\
                yearbuilt, taxamount, fips, propertylandusetypeid\
                FROM properties_2017\
                WHERE propertylandusetypeid = 261'
        connection = f'mysql+pymysql://{user}:{password}@{host}/{database}'
        zillow = pd.read_sql(query, connection)

        # Null values determined to not be worth altering as all are > 1%
        zillow = zillow.dropna()

        # rename columns
        zillow = zillow.rename(columns={'fips':'county'})
        zillow['county'] = zillow['county'].map({6037:'LA',6059:'Orange',6111:'Venture'})

        # changing value types
        zillow['yearbuilt'] = zillow['yearbuilt'].astype(int)
        zillow['bedroomcnt'] = zillow['bedroomcnt'].astype(int)
        # removing duplicate columns
        zillow = zillow.loc[:,~zillow.columns.duplicated()].copy()
        
        # normalizing numerical data
        # EXAMPLE: telco['total_charges'] = telco['total_charges'].str.replace(' ','0').astype('float')

        # additional columns for visualization
        
        # engineered features

        
        
        # dropping extra columns
        zillow = zillow.drop(columns='propertylandusetypeid')
        # restoring 'drop_first' column for contract_type as it is desired to specify just this value type (without deducting)

        # lowering all column names
        zillow.columns = map(str.lower,zillow.columns)
        zillow.columns = zillow.columns.str.replace(' ','_')

        # cache data
        zillow.to_csv('zillow.csv')
    return zillow
    
### To get single family homes, no query ###
    #single_family = zillow[zillow['propertylandusetypeid'] == 261]
    #single_family

def splitter(df,target='taxvaluedollarcnt', stratify=None):
    '''
    Returns
    Train, Validate, Test from SKLearn
    Sizes are 60% Train, 20% Validate, 20% Test
    '''
    train, test = train_test_split(df, test_size=.2, random_state=4343, stratify=stratify)

    train, validate = train_test_split(train, test_size=.2, random_state=4343, stratify=stratify)
    print(f'Dataframe: {df.shape}', '100%')
    print(f'Train: {train.shape}', '| ~60%')
    print(f'Validate: {validate.shape}', '| ~20%')
    print(f'Test: {test.shape}','| ~20%')

    return train, validate, test



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

### SCALER ###
def ScaledTrain(x_train, x_validate, x_test, linear=True):
    '''
    Produces x_train scaled with each respective style, will utilize all unless specificied otherwise.

    Arguments: df= desired data frame, Linear= True or False

    Returns: 6 or 2 arrays that would need to be assigned
    '''
    # Check for linear keyword argument to choose how to scale.
    if linear==True:
        mmscaler = MinMaxScaler()
        nscaler = StandardScaler()
        rscaler = RobustScaler()
        
        # change scaler to be kwarg to reduce output -- , scaler=

        # stored variables for associated fit x_train & transformed x_train & x_validate
        # train
        train_scaled_MinMax = mmscaler.fit_transform(x_train)
        train_scaled_S = nscaler.fit_transform(x_train)
        train_scaled_R = rscaler.fit_transform(x_train)
        # validate
        validate_scaled_MinMax = mmscaler.transform(x_validate)
        validate_scaled_S = nscaler.transform(x_validate)
        validate_scaled_R = rscaler.transform(x_validate)
        test_scaled_MinMax = mmscaler.transform(x_test)
        test_scaled_S = nscaler.transform(x_test)
        test_scaled_R = rscaler.transform(x_test)
        return\
              train_scaled_MinMax, train_scaled_S, train_scaled_R, validate_scaled_MinMax, validate_scaled_S, validate_scaled_R
    
    # Non Linear Scaler        
    else:
        qscaler = QuantileTransformer()
        # train
        train_scaled_Quant = qscaler.fit_transform(x_train)
        # validate
        validate_scaled_Quant = qscaler.transform(x_validate)
        return\
              train_scaled_Quant, validate_scaled_Quant
