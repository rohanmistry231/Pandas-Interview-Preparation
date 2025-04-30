import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [1. Introduction to Feature Engineering for ML]
# Learn Pandas techniques for creating ML features.
# Covers new feature creation, normalization, and datetime handling.

print("Pandas version:", pd.__version__)

# %% [2. Creating New Features]
# Create synthetic dataset.
np.random.seed(42)
data = {
    'length': np.random.normal(10, 2, 100),
    'width': np.random.normal(5, 1, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
}
df = pd.DataFrame(data)

# Create ratio feature
df['length_width_ratio'] = df['length'] / df['width']
print("\nDataFrame with Ratio Feature (first 3 rows):\n", df.head(3))

# Create binned feature
df['length_bin'] = pd.cut(df['length'], bins=3, labels=['short', 'medium', 'long'])
print("\nDataFrame with Binned Feature (first 3 rows):\n", df[['length', 'length_bin']].head(3))

# %% [3. Normalizing/Standardizing Features]
# Standardize length and width
df['length_std'] = (df['length'] - df['length'].mean()) / df['length'].std()
df['width_std'] = (df['width'] - df['width'].mean()) / df['width'].std()
print("\nStandardized Features (first 3 rows):\n", df[['length_std', 'width_std']].head(3))

# Normalize to [0, 1]
df['length_norm'] = (df['length'] - df['length'].min()) / (df['length'].max() - df['length'].min())
print("\nNormalized Feature (first 3 rows):\n", df[['length_norm']].head(3))

# %% [4. Handling Datetime Data]
# Create datetime dataset.
dates = pd.date_range('2023-01-01', periods=100, freq='D')
df_time = pd.DataFrame({
    'date': dates,
    'sales': np.random.normal(1000, 200, 100)
})
df_time['date'] = pd.to_datetime(df_time['date'])
df_time['month'] = df_time['date'].dt.month
df_time['day_of_week'] = df_time['date'].dt.day_name()
print("\nDatetime DataFrame (first 3 rows):\n", df_time.head(3))

# %% [5. Visualizing Features]
# Visualize standardized features
plt.figure(figsize=(8, 4))
plt.hist(df['length'], bins=20, alpha=0.5, label='Original')
plt.hist(df['length_std'], bins=20, alpha=0.5, label='Standardized')
plt.title('Length: Original vs. Standardized')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('feature_engineering_histogram.png')
plt.close()

# %% [6. Practical ML Application]
# Engineer features for ML.
np.random.seed(42)
ml_data = {
    'price': np.random.normal(50, 10, 100),
    'quantity': np.random.randint(1, 10, 100),
    'timestamp': pd.date_range('2023-01-01', periods=100, freq='H')
}
ml_df = pd.DataFrame(ml_data)
ml_df['total_value'] = ml_df['price'] * ml_df['quantity']
ml_df['timestamp'] = pd.to_datetime(ml_df['timestamp'])
ml_df['hour'] = ml_df['timestamp'].dt.hour
ml_df['price_std'] = (ml_df['price'] - ml_df['price'].mean()) / ml_df['price'].std()
print("\nML Engineered DataFrame (first 3 rows):\n", ml_df.head(3))

# %% [7. Interview Scenario: Feature Engineering]
# Discuss feature engineering for ML.
print("\nInterview Scenario: Feature Engineering")
print("Q: How would you create new features for an ML model in Pandas?")
print("A: Compute ratios, bins, or datetime attributes (e.g., df['ratio'] = df['col1'] / df['col2']).")
print("Key: New features capture relationships for better model performance.")
print("Example: df['month'] = pd.to_datetime(df['date']).dt.month")