import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('features.csv', index_col=0, parse_dates=True)
target = 'DEMAND_MET_MW'

feature_cols = ['lag_1','lag_4','lag_96','lag_192','roll_mean_4','roll_std_4',
                 'roll_mean_96','hour_sin','hour_cos','dow_sin','dow_cos',
                 'is_weekend','is_holiday']

X = df[feature_cols]
y = df[target]

# Chronological split: last 5 days as test set
split_date = df.index.max() - pd.Timedelta(days=5)
X_train, X_test = X[df.index <= split_date], X[df.index > split_date]
y_train, y_test = y[df.index <= split_date], y[df.index > split_date]

print(f"Train: {X_train.shape[0]} rows ({X_train.index.min()} to {X_train.index.max()})")
print(f"Test:  {X_test.shape[0]} rows ({X_test.index.min()} to {X_test.index.max()})")

# Scale for MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}
preds = {}

# 1. Naive baseline: predict = same time yesterday (lag_96)
naive_pred = X_test['lag_96']
preds['Naive (t-96)'] = naive_pred.values

# 2. Linear Regression
lr = LinearRegression().fit(X_train, y_train)
preds['Linear Regression'] = lr.predict(X_test)

# 3. Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
preds['Random Forest'] = rf.predict(X_test)

# 4. Gradient Boosting (XGBoost-equivalent, no external lib needed)
gb = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)
preds['Gradient Boosting'] = gb.predict(X_test)

# 5. MLP (shallow neural net, stand-in for DL until Keras/LSTM run in Colab)
mlp = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=2000, random_state=42, early_stopping=True)
mlp.fit(X_train_scaled, y_train)
preds['MLP (Neural Net)'] = mlp.predict(X_test_scaled)

# Evaluate
rows = []
for name, p in preds.items():
    mae = mean_absolute_error(y_test, p)
    rmse = np.sqrt(mean_squared_error(y_test, p))
    mape = np.mean(np.abs((y_test.values - p) / y_test.values)) * 100
    r2 = r2_score(y_test, p)
    rows.append([name, mae, rmse, mape, r2])

results_df = pd.DataFrame(rows, columns=['Model','MAE','RMSE','MAPE(%)','R2'])
results_df = results_df.sort_values('RMSE')
print()
print(results_df.to_string(index=False))
results_df.to_csv('model_comparison.csv', index=False)

# Save predictions for plotting
pred_df = pd.DataFrame(preds, index=y_test.index)
pred_df['Actual'] = y_test.values 
pred_df.to_csv('predictions.csv')

# feature importance from RF
fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print()
print("Random Forest feature importances:")
print(fi)
