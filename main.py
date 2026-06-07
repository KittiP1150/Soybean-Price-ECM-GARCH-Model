import os
import pandas as pd
from datetime import datetime
from src.data_loader import fetchAndSave, cleanAndSave
from src.ecm_model import extract_spread
from src.model import calculate_conditional_volatility
from src.trade_signal import generate_adaptive_signals
from src.backtest import run_backtest
import traceback

def main():
    
    #config
    start_date = "2023-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    #do
    try:
        fetchAndSave(start_date=start_date, end_date=end_date)
        df_clean = cleanAndSave()

        y_col = 'soybean_close'
        x_col = 'soybean_meal_close'
        
        hedge_ratio, spread, ols_model = extract_spread(df_clean, y_col, x_col)
        garch_vol = calculate_conditional_volatility(spread)
        signals_df = generate_adaptive_signals(spread, garch_vol, threshold=2.0)
        final_df = pd.concat([df_clean[[y_col, x_col]], signals_df], axis=1)
        backtest_results = run_backtest(final_df, hedge_ratio)
        
        output_path = "data/processed/backtest_results.csv"
        backtest_results.to_csv(output_path)
        active_trades = final_df[final_df['Position'] != 0]
    except Exception as e:
        print("error")
        print(f"error : {e}")
        traceback.print_exc()
        
if __name__ == "__main__":
    main()