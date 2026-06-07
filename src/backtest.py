import pandas as pd
import numpy as np

def run_backtest(df_signals, hedge_ratio, transaction_fee=0.001):
    """
    Trading Simulation
    df_signals: DataFrame price + Z-Score
    """
    bt = df_signals.copy()
    
    current_state = 0 # 0 = Nothing, 1 = Long Spead, -1 = Short Spread
    states = []
    
    for z in bt['Dynamic_Z']:
        if current_state == 0:
            if z < -2.0:
                current_state = 1
            elif z > 2.0:
                current_state = -1
                
        elif current_state == 1 and z >= 0:
            current_state = 0
        elif current_state == -1 and z <= 0:
            current_state = 0
            
        states.append(current_state)
        
    bt['Holding_State'] = states
    bt['Trade_Signal'] = bt['Holding_State'].shift(1).fillna(0)
    
    bt['Ret_ZS'] = bt['soybean_close'].pct_change()
    bt['Ret_ZM'] = bt['soybean_meal_close'].pct_change()
    bt['Spread_Return'] = bt['Ret_ZS'] - (hedge_ratio * bt['Ret_ZM'])
    
    bt['Strategy_Return'] = bt['Trade_Signal'] * bt['Spread_Return']
    bt['Cumulative_Return'] = (1 + bt['Strategy_Return']).cumprod()
    
    #transaction fee
    bt['Position_Change'] = bt['Trade_Signal'].diff().abs().fillna(0)
    bt['Transaction_Cost'] = bt['Position_Change'] * (transaction_fee * 2)
    bt['Strategy_Return'] = (bt['Trade_Signal'] * bt['Spread_Return']) - bt['Transaction_Cost']
    bt['Cumulative_Return'] = (1 + bt['Strategy_Return']).cumprod()
    #---------performance---------
    # Total Return
    total_return = bt['Cumulative_Return'].iloc[-1] - 1
    
    # Annualized Return
    # assume 252 days/years
    trading_days = len(bt)
    annualized_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
    
    # Annualized Volatility
    daily_volatility = bt['Strategy_Return'].std()
    annualized_volatility = daily_volatility * np.sqrt(252)
    
    # Sharpe Ratio (Assume Risk-free rate = 0%)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    
    # Maximum Drawdown
    running_max = bt['Cumulative_Return'].cummax()
    drawdown = (bt['Cumulative_Return'] - running_max) / running_max
    max_drawdown = drawdown.min()
    
    #Conclude
    print("\n" + "="*30)
    print(f"Total Return:       {total_return*100:8.2f} %")
    print(f"Annualized Return:  {annualized_return*100:8.2f} %")
    print(f"Max Drawdown:       {max_drawdown*100:8.2f} %")
    print(f"Sharpe Ratio:       {sharpe_ratio:8.2f}")
    print("="*30 + "\n")
    
    return bt