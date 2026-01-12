# Read processed excel
import pandas as pd
df = pd.read_csv(r"D:\DATA SCIENCE\SCHOOL PROJECTS\CRYPTO TRADER\data\processed\BTCUSDT_1d_processed.csv")


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
# Backtesting engine
INITIAL_CAPITAL = 10_000

def backtest(prices, signals):
    cash = INITIAL_CAPITAL
    position = 0

    for i in range(len(signals)):
        price = prices.iloc[i]

        if signals[i] == 2 and cash > 0:        # BUY
            position = cash / price
            cash = 0
        elif signals[i] == 0 and position > 0:  # SELL
            cash = position * price
            position = 0

    return cash + position * prices.iloc[-1]

# LOGISTIC REGRESSION
#  Imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Train / Validation / Test split (TIME-BASED)
train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)
test_start = train_size + val_size

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

X_test = X.iloc[test_start:]
y_test = y.iloc[test_start:]

# Prices needed for backtesting
test_prices = df.iloc[test_start:]["close"]

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Backtesting engine
INITIAL_CAPITAL = 10_000

def backtest(prices, signals):
    cash = INITIAL_CAPITAL
    position = 0

    for i in range(len(signals)):
        price = prices.iloc[i]

        if signals[i] == 2 and cash > 0:        # BUY
            position = cash / price
            cash = 0
        elif signals[i] == 0 and position > 0:  # SELL
            cash = position * price
            position = 0

    return cash + position * prices.iloc[-1]


# Train Logistic Regression
log_reg = LogisticRegression(
    max_iter=1000,
    multi_class="multinomial"
)

log_reg.fit(X_train, y_train)

# Predict
preds_lr = log_reg.predict(X_test)

#  Evaluation
print("\n--- Classification Metrics ---")
print("Accuracy:", accuracy_score(y_test, preds_lr))
print("Macro F1:", f1_score(y_test, preds_lr, average="macro"))
print(classification_report(y_test, preds_lr, target_names=["SELL", "HOLD", "BUY"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds_lr))

# Backtesting
final_lr = backtest(test_prices, preds_lr)

buy_hold = INITIAL_CAPITAL * (test_prices.iloc[-1] / test_prices.iloc[0])
random_signals = np.random.choice([0, 1, 2], size=len(test_prices))
random_final = backtest(test_prices, random_signals)

print("\n--- Backtest Results ---")
print("Logistic Regression Final Value:", round(final_lr, 2))
print("Buy & Hold Final Value:", round(buy_hold, 2))
print("Random Strategy Final Value:", round(random_final, 2))

# RANDOM FOREST
# Imports
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predict (TEST SET ONLY — for fair comparison)
rf_preds = rf_model.predict(X_test)

# Classification Evaluation
print("\n" + "="*60)
print("RANDOM FOREST — TEST SET METRICS")
print("="*60)

print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Macro F1:", f1_score(y_test, rf_preds, average="macro"))
print("\nClassification Report:")
print(classification_report(y_test, rf_preds, target_names=["SELL", "HOLD", "BUY"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))

# Backtesting
final_rf = backtest(test_prices, rf_preds)

buy_hold = INITIAL_CAPITAL * (test_prices.iloc[-1] / test_prices.iloc[0])
random_signals = np.random.choice([0, 1, 2], size=len(test_prices))
random_final = backtest(test_prices, random_signals)

print("\n" + "="*60)
print("RANDOM FOREST — BACKTEST RESULTS")
print("="*60)
print("Random Forest Final Value:", round(final_rf, 2))
print("Buy & Hold Final Value:", round(buy_hold, 2))
print("Random Strategy Final Value:", round(random_final, 2))

# XG BOOST
# Imports
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Train XGBoost
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss"
)

xgb.fit(X_train, y_train)

# Predict on Test Set
preds_xgb = xgb.predict(X_test)

# Evaluation
print("\n" + "="*60)
print("XGBOOST — TEST SET METRICS")
print("="*60)

print("Accuracy:", accuracy_score(y_test, preds_xgb))
print("Macro F1:", f1_score(y_test, preds_xgb, average="macro"))
print("\nClassification Report:")
print(classification_report(y_test, preds_xgb, target_names=["SELL", "HOLD", "BUY"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds_xgb))

# Backtesting
final_xgb = backtest(test_prices, preds_xgb)

buy_hold = INITIAL_CAPITAL * (test_prices.iloc[-1] / test_prices.iloc[0])
random_signals = np.random.choice([0, 1, 2], size=len(test_prices))
random_final = backtest(test_prices, random_signals)

print("\n" + "="*60)
print("XGBOOST — BACKTEST RESULTS")
print("="*60)
print("XGBoost Final Value:", round(final_xgb, 2))
print("Buy & Hold Final Value:", round(buy_hold, 2))
print("Random Strategy Final Value:", round(random_final, 2))

# CATBOOST
# Imports
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Train CatBoost
cat = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="MultiClass",
    verbose=False
)

cat.fit(X_train, y_train)

# Predict on Test Set
preds_cat = cat.predict(X_test).flatten()  # flatten needed to match shape

# Evaluation
print("\n" + "="*60)
print("CATBOOST — TEST SET METRICS")
print("="*60)

print("Accuracy:", accuracy_score(y_test, preds_cat))
print("Macro F1:", f1_score(y_test, preds_cat, average="macro"))
print("\nClassification Report:")
print(classification_report(y_test, preds_cat, target_names=["SELL", "HOLD", "BUY"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds_cat))

# Backtesting
final_cat = backtest(test_prices, preds_cat)

buy_hold = INITIAL_CAPITAL * (test_prices.iloc[-1] / test_prices.iloc[0])
random_signals = np.random.choice([0, 1, 2], size=len(test_prices))
random_final = backtest(test_prices, random_signals)

print("\n" + "="*60)
print("CATBOOST — BACKTEST RESULTS")
print("="*60)
print("CatBoost Final Value:", round(final_cat, 2))
print("Buy & Hold Final Value:", round(buy_hold, 2))
print("Random Strategy Final Value:", round(random_final, 2))

# LIGHT GBM
# Imports
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Train LightGBM
lgbm_model = LGBMClassifier(
    n_estimators=200,
    max_depth=15,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss'
)

# Predict on Test Set
preds_lgbm = lgbm_model.predict(X_test)

# Evaluation
print("\n" + "="*60)
print("LIGHTGBM — TEST SET METRICS")
print("="*60)

print("Accuracy:", accuracy_score(y_test, preds_lgbm))
print("Macro F1:", f1_score(y_test, preds_lgbm, average="macro"))
print("\nClassification Report:")
print(classification_report(y_test, preds_lgbm, target_names=["SELL", "HOLD", "BUY"]))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds_lgbm))

# Backtesting
final_lgbm = backtest(test_prices, preds_lgbm)

buy_hold = INITIAL_CAPITAL * (test_prices.iloc[-1] / test_prices.iloc[0])
random_signals = np.random.choice([0, 1, 2], size=len(test_prices))
random_final = backtest(test_prices, random_signals)

print("\n" + "="*60)
print("LIGHTGBM — BACKTEST RESULTS")
print("="*60)
print("LightGBM Final Value:", round(final_lgbm, 2))
print("Buy & Hold Final Value:", round(buy_hold, 2))
print("Random Strategy Final Value:", round(random_final, 2))

#LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Convert tabular data to tensors
# Assuming X_train, X_test, y_train, y_test are Pandas DataFrames / Series
X_train_t = torch.tensor(X_train.values[:, np.newaxis, :], dtype=torch.float32)  # shape: (samples, 1, features)
y_train_t = torch.tensor(y_train.values, dtype=torch.long)

X_test_t = torch.tensor(X_test.values[:, np.newaxis, :], dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.long)

# Define LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=3, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Instantiate model
model = LSTMClassifier(input_dim=X_train.shape[1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Move data to device
X_test_t = X_test_t.to(device)
y_test_t = y_test_t.to(device)


# Evaluation (PyTorch only)
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    y_pred_t = torch.argmax(logits, dim=1)

# Accuracy
accuracy = (y_pred_t == y_test_t).float().mean()
print("Accuracy:", accuracy.item())

# Macro F1
num_classes = 3
f1_scores = []
for cls in range(num_classes):
    tp = ((y_pred_t == cls) & (y_test_t == cls)).sum().float()
    pred_pos = (y_pred_t == cls).sum().float()
    actual_pos = (y_test_t == cls).sum().float()
    precision = tp / (pred_pos + 1e-8)
    recall = tp / (actual_pos + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    f1_scores.append(f1)
macro_f1 = sum(f1_scores) / num_classes
print("Macro F1:", macro_f1.item())

# Confusion matrix
conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int32)
for t, p in zip(y_test_t, y_pred_t):
    conf_matrix[t, p] += 1
print("Confusion Matrix:\n", conf_matrix)

# Backtesting
y_pred_list = y_pred_t.cpu().tolist()
final_lstm = backtest(test_prices, y_pred_list)

buy_hold = INITIAL_CAPITAL * (test_prices.iloc[-1] / test_prices.iloc[0])
random_signals = np.random.choice([0,1,2], size=len(test_prices))
random_final = backtest(test_prices, random_signals)

print("\nLSTM Final Value:", round(final_lstm, 2))
print("Buy & Hold Final Value:", round(buy_hold, 2))
print("Random Strategy Final Value:", round(random_final, 2))

#### Step 8 : Serialise the model
# src/train.py or notebook cell after training
import os
import joblib

# Determine project root
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
except NameError:
    # for notebooks
    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Make sure the models folder exists
models_dir = os.path.join(project_root, "models")
os.makedirs(models_dir, exist_ok=True)

# Path to save the model
model_path = os.path.join(models_dir, "buy_sell_classifier.pkl")

# Save the trained model
# 'model' is your trained scikit-learn model
joblib.dump(model, model_path)
print(f"Model saved successfully at: {model_path}")
