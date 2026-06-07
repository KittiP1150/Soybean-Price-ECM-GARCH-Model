import pandas as pd

def generate_adaptive_signals(spread, conditional_vol, threshold=2.0):
    """
    Build Z-Score + Trade signal
    """
    
    # Z-Score = (Spread - (u = 0)) / Volatility
    z_score = spread / conditional_vol
    
    signals = pd.DataFrame(index=spread.index)
    signals['Spread'] = spread
    signals['GARCH_Vol'] = conditional_vol
    signals['Dynamic_Z'] = z_score
    
    signals['Position'] = 0 
    signals.loc[signals['Dynamic_Z'] > threshold, 'Position'] = -1 #short
    signals.loc[signals['Dynamic_Z'] < -threshold, 'Position'] = 1 #long
    
    return signals