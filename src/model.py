from arch import arch_model

def calculate_conditional_volatility(spread_series):
    """
    Spead -> GARCH(1,1)
    return Volatility
    """

    scaled_spread = spread_series * 100 #gain rise
    
    # mean=zero : L(t) ~ epsilon_t 
    am = arch_model(scaled_spread, mean='Zero', vol='Garch', p=1, q=1)
    res = am.fit(disp='off') 
    
    conditional_vol = res.conditional_volatility / 100 #gain drop
    
    return conditional_vol