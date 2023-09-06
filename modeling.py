# My py files
import env
import os
import wrangle as w
# Load Datasets
from pydataset import data
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
# Modeling: Preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler, RobustScaler
# Modeling: Feature Selection
from sklearn.feature_selection import SelectKBest, f_regression, RFE
# Modeling
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import LassoLars
from sklearn.linear_model import TweedieRegressor
    # Linear: Polynomial
from sklearn.preprocessing import PolynomialFeatures
# Modeling: Selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# Metrics
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score

def regression_metrics(y_train= y_train, predictions= predictions):
    '''
    Calculates regression model metrics using SKLearn's RMSE and R2.
    '''
    
    rmse = mean_squared_error(y_train, predictions, squared = False)
    r2 = r2_score(y_train, predictions)
    return rmse, r2

def OLS(x_train = x_train_scaled, y_train = y_train, x_validate):
    '''
    Also returns validate set predictions in second return.
    '''

    lr1 = lr()
    lr1.fit(x_train, y_train)
    predictions = lr1.predict(x_train)
    val_predictions = lr1.predict(x_validate)
    rmse, r2 = regression_metrics(y_train, predictions)
    intercept = lr1.intercept_
    coef = lr1.coef_[0]
    print(f'Ordinary Least Squares, RMSE: {rmse}')
    print(f'Ordinary Least Squares, R2: {r2}')
    print(f'Intercept: {intercept}')
    print(f'Coefficient: {coef}')
    return predictions, val_predictions

def Lars(x_train = x_train_scaled, y_train = y_train, x_validate):
    '''
    Also returns validate set predictions in second return.
    '''

    lars = LassoLars(alpha=0)
    lars.fit(x_train, y_train)
    predictions = lars.predict(x_train)
    val_predictions = lars.predict(x_validate)
    rmse, r2 = regression_metrics(y_train, predictions)
    intercept = lars.intercept_
    coef = lars.coef_
    print(f'Ordinary Least Squares, RMSE: {rmse}')
    print(f'Ordinary Least Squares, R2: {r2}')
    print(f'Intercept: {intercept}')
    print(f'Coefficient: {coef}')   
    return predictions, val_predictions

def Polynom(x_train = x_train_scaled, y_train = y_train, x_validate):
    '''
    Also returns validate set predictions in second return.
    '''

    pr = lr()
    pr.fit(x_train, y_train)
    predictions = pr.predict(x_train)
    val_predictions = pr.predict(x_validate)
    rmse, r2 = regression_metrics(y_train, predictions)
    intercept = pr.intercept_
    coef = pr.coef_
    print(f'Ordinary Least Squares, RMSE: {rmse}')
    print(f'Ordinary Least Squares, R2: {r2}')
    print(f'Intercept: {intercept}')
    print(f'Coefficient: {coef}')   
    return predictions, val_predictions

def GLM(x_train = x_train_scaled, y_train = y_train, power=2, x_validate):
    '''
    Also returns validate set predictions in second return.
    '''

    glm = TweedieRegressor(power = power)
    glm.fit(x_train, y_train)
    predictions = glm.predict(x_train)
    val_predictions = glm.predicted(x_validate)
    rmse, r2 = regression_metrics(y_train, predictions)
    intercept = glm.intercept_
    coef = glm.coef_
    print(f'Ordinary Least Squares, RMSE: {rmse}')
    print(f'Ordinary Least Squares, R2: {r2}')
    print(f'Intercept: {intercept}')
    print(f'Coefficient: {coef}')   
    return predictions, val_predictions




def rfe(x_train= x_train_scaled, y_train= y_train, k= 2):
    # create the rfe object, indicating the ML object (lr) and the number of features I want to end up with. 
    rfe = RFE(lr, n_features_to_select=n_features)

    # fit the data using RFE
    rfe.fit(x_train,y_train)  

    # get the mask of the columns selected
    feature_mask = rfe.support_

    # get list of the column names. 
    rfe_feature = x_train.iloc[:,feature_mask].columns.tolist()

    return rfe_feature

def select_kbest(x_train= x_train_scaled, y_train= y_train, k= 2):

    # parameters: f_regression stats test, give me all features - normally in
    f_selector = SelectKBest(f_regression, k=k)#k='all')
    # find the all X's correlated with y
    f_selector.fit(x_train, y_train)

    # boolean mask of whether the column was selected or not
    feature_mask = f_selector.get_support()

    # get list of top K features. 
    f_feature = x_train.iloc[:,feature_mask].columns.tolist()

    return f_feature