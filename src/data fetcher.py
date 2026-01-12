#Step 1: Fetch Data From Binance

# import libraries
import requests
import pandas as pd

# Fetch the data
def fetch_binance(symbol="BTCUSDT", interval="1d", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data)
    df.columns = ["open_time","open","high","low","close","volume",
                  "close_time","quote_asset_volume","num_trades",
                  "taker_base_volume","taker_quote_volume","ignore"]
    return df
df = fetch_binance()
print(df.head())

# Save the raw data in excel
df.to_csv(r"D:\DATA SCIENCE\SCHOOL PROJECTS\MACHINE LEARNING\data\raw\BTCUSDT_1d.csv", index=False)


#Step 2: Data Cleaning & Basic Processing

# Convert time stamps to datetime
df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

# Convert numeric columns to float
cols = ['open','high','low','close','volume','quote_asset_volume','taker_base_volume','taker_quote_volume']
df[cols] = df[cols].astype(float)

# Drop unused column
df = df.drop("ignore",axis=1)

# Sort by time , chronological order
df = df.sort_values('open_time').reset_index(drop=True)

# check for NaN
df.isnull().sum()

# check for duplicate rows
duplicate_rows = df[df.duplicated()]
duplicate_rows


# Returns

# Returns
# Daily & weekly returns - [Capture Momentum] return of a financial security during regular trading hours(1 day and 7 days)
df['close_pct_1d'] = df['close'].pct_change(1)
df['close_pct_7d'] = df['close'].pct_change(7)

# Rolling volatility - [Capture Market Activity/ Risk] Calculates the volatility of an asset's price movements over a specified period.(It measures the degree of variation in the price series over time)
df['vol_7d'] = df['close_pct_1d'].rolling(7).std()
df['vol_30d'] = df['close_pct_1d'].rolling(30).std()

# Save the processed excel
df.to_csv(r"D:\DATA SCIENCE\SCHOOL PROJECTS\MACHINE LEARNING\data\processed/BTCUSDT_1d_processed.csv", index=False)

print(df.head())
print(df.info())
print(df.describe())


