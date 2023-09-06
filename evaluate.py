import env
import os
import wrangle as w
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
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import GridSearchCV
# Metrics
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score


def plot_residuals(x_train, y_train, yhat, baseline):
    '''
    '''

    residuals = x_train - yhat
    residuals_baseline = x_train - baseline

    plt.scatter(x = x_train, y = y_train)
    plt.plot(x = residuals, y = y_train)

    return

def regression_errors(x_train, y_train, yhat, baseline='mean'):
    '''
    '''
    if baseline == 'mean':
        baseline = x_train.mean()
    else:
        baseline = x_train.median()
    
    residuals = x_train - yhat
    residuals_baseline = x_train - baseline

    
    SSE =  (residuals ** 2).sum()
    SSE_baseline = (residuals_baseline ** 2).sum()
    
    print('SSE =', "{:.1f}".format(SSE))
    print("SSE Baseline =", "{:.1f}".format(SSE_baseline))
    print('~~ ~~ ~~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~~ ~~ ~~')

    MSE = SSE /len(x_train)
    MSE_baseline = SSE_baseline/len(x_train)

    print(f'MSE = {MSE:.1f}')
    print(f"MSE baseline = {MSE_baseline:.1f}")
    print('~~ ~~ ~~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~~ ~~ ~~')

    ESS = (([yhat] - y_train.mean())**2).sum()
    ESS_baseline = (([baseline] - y_train.mean())**2).sum()
    print("ESS = ", "{:.1f}".format(ESS))
    print("ESS baseline = ", "{:.1f}".format(ESS_baseline))
    print('~~ ~~ ~~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~~ ~~ ~~')

    RMSE = MSE ** .5
    RMSE_baseline = MSE_baseline ** .5
    print("RMSE = ", "{:.1f}".format(RMSE))
    print("RMSE baseline = ", "{:.1f}".format(RMSE_baseline))
    print('~~ ~~ ~~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~~ ~~ ~~')
    return SSE, SSE_baseline, MSE, MSE_baseline, ESS, ESS_baseline, RMSE, RMSE_baseline


def better_than_baseline(y,yhat):
    SSE, ESS, TSS, MSS, RMSE = regression_errors

