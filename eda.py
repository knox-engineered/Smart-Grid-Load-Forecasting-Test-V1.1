import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('nldc_combined.csv', index_col=0, parse_dates=True)

# 1. Full month load curve
plt.figure(figsize=(14,4))
plt.plot(df.index, df['DEMAND_MET_MW'], linewidth=0.8)
plt.title('All-India Demand Met — August 2026 (15-min resolution)')
plt.ylabel('MW'); plt.xlabel('Date')
plt.tight_layout()
plt.savefig('plots/01_full_month_load.png', dpi=110)
plt.close()

# 2. One week zoom to show daily seasonality
week = df.loc['2026-08-03':'2026-08-09']
plt.figure(figsize=(14,4))
plt.plot(week.index, week['DEMAND_MET_MW'])
plt.title('One Week Zoom (Aug 3-9) — Daily Seasonality')
plt.ylabel('MW'); plt.xlabel('Date')
plt.tight_layout()
plt.savefig('plots/02_week_zoom.png', dpi=110)
plt.close()

# 3. Average daily profile (hour of day pattern)
df['hour'] = df.index.hour + df.index.minute/60
hourly_profile = df.groupby('hour')['DEMAND_MET_MW'].mean()
plt.figure(figsize=(10,4))
plt.plot(hourly_profile.index, hourly_profile.values, marker='o', markersize=3)
plt.title('Average Daily Load Profile (Aug 2026)')
plt.xlabel('Hour of Day'); plt.ylabel('Avg MW')
plt.tight_layout()
plt.savefig('plots/03_avg_daily_profile.png', dpi=110)
plt.close()

# 4. Weekday vs weekend profile
df['dow'] = df.index.dayofweek  # 0=Mon
df['is_weekend'] = df['dow'].isin([5,6])
wk = df[~df['is_weekend']].groupby('hour')['DEMAND_MET_MW'].mean()
we = df[df['is_weekend']].groupby('hour')['DEMAND_MET_MW'].mean()
plt.figure(figsize=(10,4))
plt.plot(wk.index, wk.values, label='Weekday avg')
plt.plot(we.index, we.values, label='Weekend avg')
plt.legend(); plt.title('Weekday vs Weekend Load Profile')
plt.xlabel('Hour of Day'); plt.ylabel('Avg MW')
plt.tight_layout()
plt.savefig('plots/04_weekday_vs_weekend.png', dpi=110)
plt.close()

# 5. Boxplot by day of week
plt.figure(figsize=(10,4))
days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
data_by_dow = [df[df['dow']==i]['DEMAND_MET_MW'].values for i in range(7)]
plt.boxplot(data_by_dow, label=days)
plt.title('Demand Distribution by Day of Week')
plt.ylabel('MW')
plt.tight_layout()
plt.savefig('plots/05_boxplot_dow.png', dpi=110)
plt.close()

# 6. Correlation heatmap (numeric cols)
num_cols = ['FREQUENCY_Hz','DEMAND_MET_MW','NUCLEAR_MW','WIND_MW','SOLAR_MW',
            'HYDRO_MW','GAS_MW','THERMAL_MW','STORAGE_MW','NET_DEMAND_MET_MW',
            'TOTAL_GENERATION_MW']
corr = df[num_cols].corr()
plt.figure(figsize=(9,7))
im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im)
plt.xticks(range(len(num_cols)), num_cols, rotation=90)
plt.yticks(range(len(num_cols)), num_cols)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('plots/06_correlation_heatmap.png', dpi=110)
plt.close()

print("EDA plots saved.")
print()
print("Correlation of each feature with DEMAND_MET_MW:")
print(corr['DEMAND_MET_MW'].sort_values(ascending=False))
