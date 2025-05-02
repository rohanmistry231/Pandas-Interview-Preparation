import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Initial Data Cleaning]
# Learn how to clean Pandas DataFrames for ML.
# Covers handling missing values, duplicates, and data type conversions.

print("Pandas version:", pd.__version__)

# %% [2. Handling Missing Values]
# Create DataFrame with missing values.
if load_iris:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
else:
    data = {
        'sepal length (cm)': [5.1, 4.9, 7.0],
        'sepal width (cm)': [3.5, 3.0, 3.2],
        'petal length (cm)': [1.4, 1.4, 4.7],
        'petal width (cm)': [0.2, 0.2, 1.4]
    }
    df = pd.DataFrame(data)

# Introduce missing values
np.random.seed(42)
mask = np.random.rand(*df.shape) < 0.1
df[mask] = np.nan

print("\nDataFrame with Missing Values (first 3 rows):\n", df.head(3))
print("\nMissing Values Count:\n", df.isna().sum())

# Fill missing values with mean
df_filled = df.fillna(df.mean())
print("\nFilled Missing Values (first 3 rows):\n", df_filled.head(3))

# Drop rows with missing values
df_dropped = df.dropna()
print("\nDropped Missing Rows Shape:", df_dropped.shape)

# %% [3. Dropping Duplicates]
# Introduce duplicates
df_duplicates = pd.concat([df, df.iloc[:5]], ignore_index=True)
print("\nDataFrame with Duplicates Shape:", df_duplicates.shape)

# Drop duplicates
df_no_duplicates = df_duplicates.drop_duplicates()
print("\nDataFrame after Dropping Duplicates Shape:", df_no_duplicates.shape)

# %% [4. Data Type Conversions]
# Convert float to int (after filling missing values)
df_filled['sepal length (cm)'] = df_filled['sepal length (cm)'].astype(int)
print("\nData Types after Conversion:\n", df_filled.dtypes)

# Convert to category for ML encoding
df_filled['petal_category'] = pd.cut(df_filled['petal length (cm)'], bins=3, labels=['short', 'medium', 'long'])
print("\nDataFrame with Category Column (first 3 rows):\n", df_filled[['petal length (cm)', 'petal_category']].head(3))

# %% [5. Visualizing Cleaning]
# Visualize missing values before/after.
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
df.isna().sum().plot(kind='bar', color='red')
plt.title('Missing Values Before')
plt.subplot(1, 2, 2)
df_filled.isna().sum().plot(kind='bar', color='green')
plt.title('Missing Values After')
plt.tight_layout()
plt.savefig('initial_data_cleaning_missing.png')

# %% [6. Practical ML Application]
# Clean a synthetic ML dataset.
np.random.seed(42)
ml_data = {
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100) * 100,
    'target': np.random.randint(0, 2, 100)
}
ml_df = pd.DataFrame(ml_data)
ml_df.iloc[::10, 0] = np.nan  # Introduce missing values
ml_df_cleaned = ml_df.fillna(ml_df.mean()).drop_duplicates()
ml_df_cleaned['feature2'] = ml_df_cleaned['feature2'].astype(int)
print("\nCleaned ML DataFrame (first 3 rows):\n", ml_df_cleaned.head(3))

# %% [7. Interview Scenario: Missing Values]
# Discuss handling missing values for ML.
print("\nInterview Scenario: Missing Values")
print("Q: How would you handle missing values in a DataFrame for ML?")
print("A: Use df.fillna(mean) for numeric data or df.dropna for small datasets.")
print("Key: Imputation preserves data; dropping reduces size.")
print("Example: df.fillna(df.mean()) for feature imputation.")