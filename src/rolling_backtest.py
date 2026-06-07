import os
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

def run_rolling_backtest(input_dir="data/processed", test_days=30, threshold_pct=0.005, max_vol_tolerance=0.02):
    
    features_path = os.path.join(input_dir, "features_soybean_fx.csv")
    
    try:
        df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Error: features_soybean_fx.csv not found.")
        return

    backtest_results = []
    
    total_rows = len(df)
    
    print("Training models and predicting day-by-day...")
    for i in tqdm(range(total_rows - test_days, total_rows)):
        
        train_data = df.iloc[:i]['soybean_close']
        latest_actual_price = train_data.iloc[-1]
        
        target_date = df.index[i]
        actual_price_today = df['soybean_close'].iloc[i]
        
        arima_model = ARIMA(train_data, order=(0,1,0))
        arima_fit = arima_model.fit()
        forecast_price = arima_fit.forecast(steps=1).iloc[0]

        residuals = arima_fit.resid.dropna()
        garch_model = arch_model(residuals, vol='GARCH', p=1, q=1, rescale=False, dist='t')
        garch_fit = garch_model.fit(disp='off')
        
        forecast_var = garch_fit.forecast(horizon=1).variance.iloc[-1, 0]
        forecast_vol_points = np.sqrt(forecast_var)
        
        expected_change = (forecast_price - latest_actual_price) / latest_actual_price
        forecast_vol_pct = forecast_vol_points / latest_actual_price
        
        if expected_change > threshold_pct:
            base_signal = "BUY"
        elif expected_change < -threshold_pct:
            base_signal = "SELL"
        else:
            base_signal = "HOLD"
            
        if base_signal in ["BUY", "SELL"] and forecast_vol_pct > max_vol_tolerance:
            signal = "HOLD (High Risk)"
        else:
            signal = base_signal
            
        daily_pnl = 0
        if signal == "BUY":
            daily_pnl = actual_price_today - latest_actual_price
        elif signal == "SELL":
            daily_pnl = latest_actual_price - actual_price_today
            
        backtest_results.append({
            'Date': target_date,
            'Actual_Previous': latest_actual_price,
            'Forecast_Price': forecast_price,
            'Actual_Today': actual_price_today,
            'Forecast_Vol(%)': forecast_vol_pct,
            'Signal': signal,
            'PnL_USD': daily_pnl
        })

    df_results = pd.DataFrame(backtest_results)
    df_results.set_index('Date', inplace=True)
    df_results['Cumulative_PnL'] = df_results['PnL_USD'].cumsum()
    
    output_path = os.path.join(input_dir, "arima_garch_backtest.csv")
    df_results.to_csv(output_path)
    
    total_trades = len(df_results[df_results['Signal'].isin(['BUY', 'SELL'])])
    win_trades = len(df_results[df_results['PnL_USD'] > 0])
    loss_trades = len(df_results[df_results['PnL_USD'] < 0])
    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
    total_net_profit = df_results['Cumulative_PnL'].iloc[-1]
    
    print("\n" + "="*50)
    print("ARIMA-GARCH Backtest Results")
    print("="*50)
    print(f"Test Period       : {test_days} Days")
    print(f"Total Trades Taken: {total_trades}")
    print(f"Winning Trades    : {win_trades}")
    print(f"Losing Trades     : {loss_trades}")
    print(f"Win Rate          : {win_rate:.2f}%")
    print(f"Net Profit (USD)  : ${total_net_profit:.2f}")
    print("="*50)
    print(f"Details saved to: {output_path}")

if __name__ == "__main__":
    run_rolling_backtest(test_days=365)