import pandas as pd
import numpy as np

df = pd.read_csv('nldc_combined.csv', index_col=0, parse_dates=True)
target = 'DEMAND_MET_MW'

df = df[[target]].copy()

# --- Lag features ---
df['lag_1'] = df[target].shift(1)        # previous 15-min block
df['lag_4'] = df[target].shift(4)        # 1 hour ago
df['lag_96'] = df[target].shift(96)      # same time yesterday
df['lag_192'] = df[target].shift(192)    # same time, 2 days ago

# --- Rolling stats (based on past values only, no leakage) ---
df['roll_mean_4'] = df[target].shift(1).rolling(4).mean()    # last 1hr avg
df['roll_std_4']  = df[target].shift(1).rolling(4).std()
df['roll_mean_96'] = df[target].shift(1).rolling(96).mean()  # last 24hr avg

# --- Calendar features ---
df['hour'] = df.index.hour
df['minute'] = df.index.minute
df['block_of_day'] = df.index.hour * 4 + df.index.minute // 15  # 0-95
df['dow'] = df.index.dayofweek
df['is_weekend'] = (df['dow'] >= 5).astype(int)

# India Independence Day fell in this window
df['is_holiday'] = (df.index.date == pd.Timestamp('2026-08-15').date()).astype(int)

# --- Cyclical encoding ---
df['hour_sin'] = np.sin(2*np.pi*df['block_of_day']/96)
df['hour_cos'] = np.cos(2*np.pi*df['block_of_day']/96)
df['dow_sin'] = np.sin(2*np.pi*df['dow']/7)
df['dow_cos'] = np.cos(2*np.pi*df['dow']/7)

df = df.dropna()  # drop rows with NaN from lag/rolling windows (first ~2 days)
df.to_csv('features.csv')
print("Feature dataset shape:", df.shape)
print("Date range:", df.index.min(), "to", df.index.max())
print(df.columns.tolist())
