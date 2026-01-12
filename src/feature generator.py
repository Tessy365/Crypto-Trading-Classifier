#Step 3: Feature Engineering

# Technical Indicators

# Read processed excel
import pandas as pd
df = pd.read_csv(r"D:\DATA SCIENCE\SCHOOL PROJECTS\MACHINE LEARNING\data\processed\BTCUSDT_1d_processed.csv")

# Simple Moving Average (SMA) :Smooths price over time to spot trends- sum of the last 20,30,50 days of close divided by 20,30,50
df["sma_20"] = df["close"].rolling(20).mean()   # 20-day SMA
df["sma_50"] = df["close"].rolling(50).mean()   # 50-day SMA
df["sma_200"] = df["close"].rolling(200).mean() # 200-day SMA

import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))

# Plot the close price
plt.plot(df.index, df["close"], label="Close Price", linewidth=1)

# Plot SMAs
plt.plot(df.index, df["sma_20"], label="SMA 20")
plt.plot(df.index, df["sma_50"], label="SMA 50")
plt.plot(df.index, df["sma_200"], label="SMA 200")

plt.title("Simple Moving Averages (20, 50, 200 Days)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# RSI (Relative Strength Index)- Shows overbought or oversold conditions,RSI ranges 0–100
# 70 → Overbought (possible Sell)
# <30 → Oversold (possible Buy)
import ta
df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

import matplotlib.pyplot as plt

plt.figure(figsize=(14,5))

plt.plot(df.index, df["rsi"], label="RSI")

# Add horizontal lines for thresholds
plt.axhline(70, linestyle="--", label="Overbought (70)")
plt.axhline(30, linestyle="--", label="Oversold (30)")

plt.title("RSI (Relative Strength Index)")
plt.xlabel("Date")
plt.ylabel("RSI Value")
plt.legend()
plt.show()


# Moving Average Convergence Divergence (MACD) - Captures trend changes, helps identify potential ttrend reversals, momentum strength and overall direction of an asset's price.
# MACDLine=12−dayEMA−26−dayEMA
# SignalLine=9−dayEMA
# MACDHistogram=MACDLine−SignalLine
# MACD line minus signal line indicates bullish/bearish momentum.
macd_object = ta.trend.MACD(df["close"])
df["macd"] = macd_object.macd()
df["macd signal"] = macd_object.macd_signal()
df["macd diff"] = macd_object.macd_diff()
df.tail()

import matplotlib.pyplot as plt

plt.figure(figsize=(14,7))

#Plotting Close Price
plt.subplot(2,1,1)
plt.plot(df["close"],label="close Price")
plt.title("Crypto MACD")
plt.legend()

#Plottinf MACD
plt.subplot(2,1,2)
plt.plot(df["macd"],label="MACD Line",color="blue")
plt.plot(df["macd signal"], label="Signal Line", color="red")
plt.bar(df.index, df["macd diff"],label="Histogram",color="grey",alpha=0.5)
plt.legend()

plt.show()

# Bollinger Bands - Bollinger Bands show price relative to volatility
# Price near upper band → possible overbought
# Price near lower band → possible oversold
# Bollinger Bands consist of a middle band, which is a simple moving average, and two outer bands that represent standard deviations from the moving average.

from ta.volatility import BollingerBands

# Create the indicator
bb = BollingerBands(close=df["close"], window=20, window_dev=2)

# Add Bollinger Bands to the dataframe
df["bb_high"] = bb.bollinger_hband()
df["bb_mid"] = bb.bollinger_mavg()
df["bb_low"] = bb.bollinger_lband()


plt.figure(figsize=(14,6))
plt.plot(df.index, df["close"], label="Close")
plt.plot(df.index, df["bb_high"], label="BB High")
plt.plot(df.index, df["bb_mid"], label="BB Mid")
plt.plot(df.index, df["bb_low"], label="BB Low")
plt.legend()
plt.show()

# Stochastic Oscillator
# The Stochastic Oscillator compares a closing price to its price range over time. It has two lines: %K (fast) and %D (slow). These lines move between 0 and 100, showing momentum and trend strength.
# %K = 100 * (Current Close — Lowest Low) / (Highest High — Lowest Low)
# %D = 3-day SMA of %K
# The standard lookback period is 14 days, but you can adjust it.
# Values over 80, it means there may be overbuying


# Compute Stochastic Oscillator and add to df
stoch = ta.momentum.StochasticOscillator(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    window=14,
    smooth_window=3
)

df['stoch_k'] = stoch.stoch()        # Smoothed %K
df['stoch_d'] = stoch.stoch_signal() # %D line

#df["stochastic_oscillator"] = ta.momentum.StochasticOscillator(
#    high=df["high"],
#    low=df["low"],
#    close=df["close"]
#).stoch()

plt.figure(figsize=(14,6))
plt.plot(df.index, df['stoch_k'], label='%K')
plt.plot(df.index, df['stoch_d'], label='%D')
plt.axhline(80, color='red', linestyle='--', label='Overbought (80)')
plt.axhline(20, color='green', linestyle='--', label='Oversold (20)')
plt.title("Stochastic Oscillator (%K and %D)")
plt.xlabel("Date")
plt.ylabel("Value (0–100)")
plt.legend()
plt.show()

#Keep num_trades and volume-related features-  Captures market activity trends.
# Example rolling stats
df['trades_7d_avg'] = df['num_trades'].rolling(7).mean()
df['trades_30d_avg'] = df['num_trades'].rolling(30).mean()
df['volume_7d_avg'] = df['volume'].rolling(7).mean()
df['volume_30d_avg'] = df['volume'].rolling(30).mean()

# Drop rows with NaN after rolling calculations - Rolling calculations (SMA, RSI, volatility) create NaNs at the beginning.
# Drop them before feeding into ML models.

df = df.dropna().reset_index(drop=True)

# features These are the columns I want the model to see
# X - the actual data from those columns, ready for ML
features = [
    "close_pct_1d", "close_pct_7d", "vol_7d", "vol_30d",
    "rsi", "sma_20", "sma_50", "sma_200",
    "bb_high", "bb_mid", "bb_low",
    "stoch_k", "stoch_d",
    "trades_7d_avg", "trades_30d_avg",
    "volume_7d_avg", "volume_30d_avg"
]
X = df[features]


# Scale Features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Check Feature Correlation
# Helps avoid redundant features
# Can drop features that are highly correlated (>0.9)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))
sns.heatmap(df[features].corr(), annot=True, fmt=".2f")
plt.show()


