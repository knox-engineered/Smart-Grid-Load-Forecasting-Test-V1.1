"""
LSTM Load Forecasting — run this in Google Colab (TensorFlow is pre-installed there).
Upload features.csv (produced by features.py) to Colab first, then run this script/notebook.

pip install tensorflow  # only needed if not already available
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow.keras.models
import tensorflow.keras.layers
import tensorflow.keras.callbacks

# ---- 1. Load features ----
df = pd.read_csv('features.csv', index_col=0, parse_dates=True)
target = 'DEMAND_MET_MW'

feature_cols = ['lag_1', 'lag_4', 'lag_96', 'lag_192', 'roll_mean_4', 'roll_std_4',
                 'roll_mean_96', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                 'is_weekend', 'is_holiday']

# ---- 2. Scale everything (LSTM needs scaled inputs) ----
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_all = scaler_X.fit_transform(df[feature_cols])
y_all = scaler_y.fit_transform(df[[target]])

# ---- 3. Build sequences: use past `window` steps to predict the next step ----
window = 96  # past 24 hours (15-min blocks) to predict next block
X_seq, y_seq = [], []
for i in range(window, len(X_all)):
    X_seq.append(X_all[i-window:i])
    y_seq.append(y_all[i])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# ---- 4. Chronological split: last 5 days (480 blocks) as test ----
test_size = 480
X_train, X_test = X_seq[:-test_size], X_seq[-test_size:]
y_train, y_test = y_seq[:-test_size], y_seq[-test_size:]

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---- 5. Build LSTM model ----
model = tensorflow.keras.models.Sequential([
    tensorflow.keras.layers.LSTM(64, return_sequences=True, input_shape=(window, X_train.shape[2])),
    tensorflow.keras.layers.Dropout(0.2),
    tensorflow.keras.layers.LSTM(32),
    tensorflow.keras.layers.Dropout(0.2),
    tensorflow.keras.layers.Dense(16, activation='relu'),
    tensorflow.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=60,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# ---- 6. Predict and inverse-scale ----
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = r2_score(y_true, y_pred)

print(f"\nLSTM Results:")
print(f"MAE:  {mae:.2f} MW")
print(f"RMSE: {rmse:.2f} MW")
print(f"MAPE: {mape:.3f} %")
print(f"R2:   {r2:.4f}")

# ---- 7. Plot ----
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 5))
plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='LSTM Predicted')
plt.legend()
plt.title('LSTM: Actual vs Predicted Demand (Test Set)')
plt.ylabel('MW')
plt.tight_layout()
plt.savefig('lstm_actual_vs_predicted.png', dpi=110)
plt.show()

# ---- 8. Compare against your earlier classical-ML results ----
# Load model_comparison.csv (from train_models.py) and append this LSTM row
comparison = pd.read_csv('model_comparison.csv')
comparison.loc[len(comparison)] = ['LSTM', mae, rmse, mape, r2]
comparison = comparison.sort_values('RMSE')
print("\nFull model comparison:")
print(comparison.to_string(index=False))
comparison.to_csv('model_comparison_with_lstm.csv', index=False)

# ---- 9. Save per-timestamp LSTM predictions, merged into predictions.csv ----
# The test set here is the LAST `test_size` rows of features.csv (same chronological
# split used in train_models.py), so we can align it back to those exact dates.
test_dates = df.index[-test_size:]

lstm_pred_df = pd.DataFrame({
    'LSTM': y_pred.flatten()
}, index=test_dates)

try:
    existing_preds = pd.read_csv('predictions.csv', index_col=0, parse_dates=True)
    merged = existing_preds.join(lstm_pred_df, how='left')
    merged.to_csv('predictions.csv')
    print("\nMerged LSTM predictions into existing predictions.csv")
except FileNotFoundError:
    # No earlier predictions.csv found — save LSTM predictions standalone
    lstm_pred_df['Actual'] = y_true.flatten()
    lstm_pred_df.to_csv('predictions.csv')
    print("\npredictions.csv not found — created new one with LSTM + Actual only")
