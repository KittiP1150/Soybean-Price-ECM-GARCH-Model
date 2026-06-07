import numpy as np
import pandas as pd
import statsmodels.api as sm

def extract_spread(df, asset_y_col, asset_x_col):
    """
    Long term equilibrium Cointegration (OLS)
    Generate Error Term (Spread)
    """
    
    log_y = np.log(df[asset_y_col])
    log_x = np.log(df[asset_x_col])
    
    #OLS: log_y = alpha + gamma(log_x) + error
    X = sm.add_constant(log_x)
    model = sm.OLS(log_y, X).fit()
    
    hedge_ratio = model.params.iloc[1]
    mean_mu = model.params.iloc[0]
    
    #Spread
    spread = model.resid 
    
    return hedge_ratio, spread, model