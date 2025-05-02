import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Advanced Data Cleaning]
# Learn advanced Pandas techniques for cleaning ML datasets.
# Covers outlier handling, string operations, and categorical encoding.

print("Pandas version:", pd.__version__)

# %% [2. Handling Outliers]
# Load Iris dataset or synthetic data.
if load_iris:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
else:
    np.random.seed(42)
    data = {
        'sepal length (cm)': np.random.normal(5.8, 0.8, 150),
        'sepal width (cm)': np.random.normal(3.0, 0.4, 150),
        'petal length (cm)': np.random.normal(3.7, 1.8, 150),
        'petal width (cm)': np.random.normal(1.2, 0.6, 150)
    }
    df = pd.DataFrame(data)

# Introduce outliers
df.iloc[::10, 0] = df['sepal length (cm)'] * 2  # Double some values
print("\nDataFrame with Outliers (first 3 rows):\n", df.head(3))

# Detect and clip outliers using quantiles
q_low, q_high = df['sepal length (cm)'].quantile([0.05, 0.95])
df['sepal length (cm)'] = df['sepal length (cm)'].clip(lower=q_low, upper=q_high)
print("\nDataFrame after Clipping Outliers (first 3 rows):\n", df.head(3))

# %% [3. String Operations]
# Create synthetic retail dataset with strings.
np.random.seed(42)
retail_data = {
    'product': ['Laptop_123', 'Phone_AB12', 'Tablet_XY45'] * 33,
    'price': np.random.normal(500, 100, 99)
}
df_retail = pd.DataFrame(retail_data)

# Extract product type
df_retail['product_type'] = df_retail['product'].str.extract(r'(\w+)_')[0]
print("\nRetail DataFrame with Extracted Product Type (first 3 rows):\n", df_retail.head(3))

# Replace underscores
df_retail['product_clean'] = df_retail['product'].str.replace('_', '-')
print("\nRetail DataFrame with Cleaned Product Names (first 3 rows):\n", df_retail.head(3))

# %% [4. Encoding Categorical Variables]
# Encode product type
df_retail['product_type_encoded'] = df_retail['product_type'].map({'Laptop': 0, 'Phone': 1, 'Tablet': 2})
print("\nRetail DataFrame with Mapped Encoding (first 3 rows):\n", df_retail[['product_type', 'product_type_encoded']].head(3))

# One-hot encoding
df_one_hot = pd.get_dummies(df_retail, columns=['product_type'], prefix='type')
print("\nRetail DataFrame with One-Hot Encoding (first 3 rows):\n", df_one_hot.head(3))

# %% [5. Visualizing Cleaning]
# Box plot for outliers
plt.figure(figsize=(8, 4))
plt.boxplot([df_retail['price'], df_retail['price'].clip(df_retail['price'].quantile(0.05), df_retail['price'].quantile(0.95))],
            labels=['Before Clipping', 'After Clipping'])
plt.title('Price Outliers Before and After Clipping')
plt.ylabel('Price ($)')
plt.savefig('advanced_data_cleaning_boxplot.png')
plt.close()

# %% [6. Practical ML Application]
# Clean and encode a synthetic ML dataset.
np.random.seed(42)
ml_data = {
    'feature1': np.random.normal(10, 2, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'text': ['item_' + str(i) for i in range(100)]
}
ml_df = pd.DataFrame(ml_data)
ml_df.iloc[::10, 0] = ml_df['feature1'] * 3  # Introduce outliers
ml_df['feature1'] = ml_df['feature1'].clip(ml_df['feature1'].quantile(0.05), ml_df['feature1'].quantile(0.95))
ml_df['category_encoded'] = pd.get_dummies(ml_df['category'], prefix='cat').values.argmax(axis=1)
ml_df['text_clean'] = ml_df['text'].str.replace('item_', 'product_')
print("\nCleaned ML DataFrame (first 3 rows):\n", ml_df.head(3))

# %% [7. Interview Scenario: Categorical Encoding]
# Discuss encoding for ML.
print("\nInterview Scenario: Categorical Encoding")
print("Q: How would you encode categorical variables in Pandas for ML?")
print("A: Use pd.get_dummies for one-hot encoding or df['col'].map for ordinal encoding.")
print("Key: One-hot encoding avoids ordinal assumptions; mapping is memory-efficient.")
print("Example: pd.get_dummies(df, columns=['category']) for dummy variables.")