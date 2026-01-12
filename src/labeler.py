# Read processed excel
import pandas as pd
df = pd.read_csv(r"D:\DATA SCIENCE\SCHOOL PROJECTS\CRYPTO TRADER\data\processed\crypto_labeled_features.csv")


#Select Good Features only
#Step 4: Label Generation (Target Variable)

### Rule-Based Labeling (Multiclass)
### If next_day_return > +2% → BUY
### If next_day_return < –2% → SELL
### Else → HOLD  

"""
Generates Buy/Sell/Hold labels based on future returns.

    Parameters:
    - threshold_buy  : % return above which we BUY (default 2%)
    - threshold_sell : % return below which we SELL (default -2%)
    - future_period  : number of days ahead to calculate return (default 1)
    
    Labels:
    0 = SELL
    1 = HOLD
    2 = BUY

"""

df["future_return"] = df["close"].pct_change().shift(-1)
 
def label(row):
    if row["future_return"] > 0.02:
        return 2
    elif row["future_return"] < -0.02:
        return 0
    else:
        return 1
 
df["label"] = df.apply(label, axis=1)

print(df.head())

# Save the processed excel
df.to_csv(r"D:\DATA SCIENCE\SCHOOL PROJECTS\MACHINE LEARNING\data\processed/crypto_labeled_features.csv", index=False)



#Step 5: Train/Test Split
#Plotting a Feature Correlation Matrix to determine what columns to use for the model
import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = [
    'future_return', 'label', 'rsi', 'macd', 'macd_hist',
    'sma_20', 'sma_50', 'sma_200', 'volatility', 'volume'
]

#Filtering the columns in the dataframe
existing_cols = [col for col in numeric_cols if col in df.columns]

plt.figure(figsize=(12, 10))
sns.heatmap(df[existing_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.show()

# Define features and target

target_col = "label"  # column with your Buy/Sell/Hold labels
feature_cols = df.columns.difference([target_col, "future_return", "open_time", "close_time"])

X = df[feature_cols]
y = df[target_col]


# Train/Validation/Test split

train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)
test_start = train_size + val_size

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_val = X.iloc[train_size:test_start]
y_val = y.iloc[train_size:test_start]

X_test = X.iloc[test_start:]
y_test = y.iloc[test_start:]


# Print summary

print(f"Train: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
print(f"Validation: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
print(f"Test: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")

# Extract test-period prices for backtesting
test_prices = df.iloc[test_start:]["close"]

# Step 6: Model Training


