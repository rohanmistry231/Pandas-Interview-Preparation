import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [1. Introduction to Advanced Analytics]
# Learn advanced Pandas analytics for ML insights.
# Covers time-series analysis, MultiIndex/pivot operations, and statistical computations.

print("Pandas version:", pd.__version__)

# %% [2. Time-series Analysis]
# Create synthetic time-series dataset.
dates = pd.date_range('2023-01-01', periods=365, freq='D')
np.random.seed(42)
ts_df = pd.DataFrame({
    'date': dates,
    'sales': np.random.normal(1000, 200, 365)
})
ts_df.set_index('date', inplace=True)

# Resample to monthly
monthly_sales = ts_df.resample('M').mean()
print("\nMonthly Sales (first 3 rows):\n", monthly_sales.head(3))

# Rolling mean
ts_df['rolling_mean'] = ts_df['sales'].rolling(window=7).mean()
print("\nRolling Mean (first 3 rows):\n", ts_df[['sales', 'rolling_mean']].head(3))

# Exponentially weighted moving average
ts_df['ewm'] = ts_df['sales'].ewm(span=7).mean()
print("\nEWM (first 3 rows):\n", ts_df[['sales', 'ewm']].head(3))

# %% [3. MultiIndex and Pivot Operations]
# Create retail dataset with MultiIndex.
np.random.seed(42)
retail_data = {
    'store': np.random.choice(['A', 'B'], 200),
    'product': np.random.choice(['Laptop', 'Phone'], 200),
    'month': np.random.choice(['Jan', 'Feb', 'Mar'], 200),
    'sales': np.random.normal(1000, 200, 200)
}
df = pd.DataFrame(retail_data)
df_multi = df.set_index(['store', 'product'])

# Pivot operation
pivot = df.pivot_table(values='sales', index='store', columns=['month', 'product'], aggfunc='mean')
print("\nPivot Table:\n", pivot.head())

# Melt operation
melted = pd.melt(df, id_vars=['store', 'product'], value_vars=['sales'], value_name='sales_value')
print("\nMelted DataFrame (first 3 rows):\n", melted.head(3))

# %% [4. Advanced Statistical Computations]
# Correlation and covariance
corr_matrix = df[['sales']].corr()
cov_matrix = df[['sales']].cov()
print("\nCorrelation Matrix:\n", corr_matrix)
print("\nCovariance Matrix:\n", cov_matrix)

# %% [5. Visualizing Analytics]
# Visualize time-series
plt.figure(figsize=(8, 4))
plt.plot(ts_df.index, ts_df['sales'], label='Sales', alpha=0.5)
plt.plot(ts_df.index, ts_df['rolling_mean'], label='7-day Rolling Mean', color='red')
plt.plot(ts_df.index, ts_df['ewm'], label='EWM', color='green')
plt.title('Time-series Analysis: Sales')
plt.xlabel('Date')
plt.ylabel('Sales ($)')
plt.legend()
plt.savefig('advanced_analytics_timeseries.png')
plt.close()

# Visualize pivot table
plt.figure(figsize=(8, 4))
plt.imshow(pivot, cmap='coolwarm')
plt.colorbar(label='Mean Sales ($)')
plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.title('Pivot Table: Sales by Store, Month, Product')
plt.savefig('advanced_analytics_pivot.png')
plt.close()

# %% [6. Practical ML Application]
# Analyze time-series features for ML.
np.random.seed(42)
ml_ts_df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100, freq='D'),
    'feature1': np.random.normal(50, 10, 100)
})
ml_ts_df.set_index('date', inplace=True)
ml_ts_df['rolling_mean'] = ml_ts_df['feature1'].rolling(window=5).mean()
ml_ts_df['lag_1'] = ml_ts_df['feature1'].shift(1)
print("\nML Time-series Features (first 3 rows):\n", ml_ts_df.head(3))

# %% [7. Interview Scenario: Time-series]
# Discuss time-series for ML.
print("\nInterview Scenario: Time-series")
print("Q: How would you prepare time-series data for ML in Pandas?")
print("A: Use resample, rolling, or ewm to create features; add lags with shift.")
print("Key: Time-series features capture temporal patterns.")
print("Example: df['rolling'] = df['col'].rolling(window=5).mean()")