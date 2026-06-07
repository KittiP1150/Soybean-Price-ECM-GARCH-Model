import yfinance as yf
import pandas as pd
import os

def fetchAndSave(start_date, end_date, raw_dir="data/raw"):
    "pull data and save in raw"
    os.makedirs(raw_dir, exist_ok=True)
    Tickers = {
        "soybean" : "ZS=F",
        "soybean_meal": "ZM=F",
        "usd_thb" : "THB=X"
    }
    
    for name, ticker in Tickers.items():
        print(f"data : {ticker}")
        df = yf.download(tickers= ticker, start = start_date, end= end_date)
        df_ohlc = df[['Open', 'High', 'Low', 'Close']].copy()
        if isinstance(df_ohlc.columns, pd.MultiIndex):
            df_ohlc.columns = df_ohlc.columns.droplevel(1)
        df_ohlc.columns = [f'{name}_{col.lower()}' for col in df_ohlc.columns]
        
        df_ohlc.index = df_ohlc.index.tz_localize(None)
        df_ohlc.index.name = 'Date'
        file_path = os.path.join(raw_dir, f"raw_{name}.csv")
        df_ohlc.to_csv(file_path)
        
def cleanAndSave(raw_dir ="data/raw", processed_dir = "data/processed"):
    "load data + combine + dropN/A + save in processed"
    os.makedirs(processed_dir, exist_ok=True)
    soy_path = os.path.join(raw_dir, "raw_soybean.csv")
    fx_path = os.path.join(raw_dir, "raw_usd_thb.csv")
    
    df_soy = pd.read_csv(soy_path, index_col=0, parse_dates=True)
    df_meal = pd.read_csv(os.path.join(raw_dir, "raw_soybean_meal.csv"), index_col=0, parse_dates=True)
    df_fx =pd.read_csv(fx_path, index_col=0,parse_dates=True)
    
    df_merge = pd.merge(df_soy, df_meal, left_index=True, right_index=True, how='inner')
    df_merge = pd.merge(df_merge, df_fx, left_index=True, right_index=True, how='left')
    
    df_merge = df_merge.apply(pd.to_numeric, errors='coerce')
    df_clean = df_merge.ffill().dropna()
    df_clean['soybean_open'] = df_clean['soybean_open'] / 100
    df_clean['soybean_high'] = df_clean['soybean_high'] / 100
    df_clean['soybean_low'] = df_clean['soybean_low'] / 100
    df_clean['soybean_close'] = df_clean['soybean_close'] / 100
    
    file_path = os.path.join(processed_dir, "processed_soybean_fx.csv")
    df_clean.to_csv(file_path)
    
    return df_clean

    