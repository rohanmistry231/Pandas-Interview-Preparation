import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [1. Introduction to Merging and Joining]
# Learn how to combine Pandas DataFrames for ML datasets.
# Covers merge, join, concat, and handling join types.

print("Pandas version:", pd.__version__)

# %% [2. Combining Datasets]
# Create synthetic retail datasets.
np.random.seed(42)
orders = pd.DataFrame({
    'order_id': range(100),
    'customer_id': np.random.randint(1, 20, 100),
    'amount': np.random.normal(100, 20, 100)
})
customers = pd.DataFrame({
    'customer_id': range(1, 25),
    'name': ['Customer_' + str(i) for i in range(1, 25)]
})

# Merge datasets
df_merged = pd.merge(orders, customers, on='customer_id', how='inner')
print("\nMerged DataFrame (inner join, first 3 rows):\n", df_merged.head(3))

# Different join types
df_left = pd.merge(orders, customers, on='customer_id', how='left')
df_outer = pd.merge(orders, customers, on='customer_id', how='outer')
print("\nLeft Join Shape:", df_left.shape)
print("Outer Join Shape:", df_outer.shape)

# %% [3. Using Join]
# Join with index
orders_indexed = orders.set_index('customer_id')
customers_indexed = customers.set_index('customer_id')
df_joined = orders_indexed.join(customers_indexed, how='inner').reset_index()
print("\nJoined DataFrame (first 3 rows):\n", df_joined.head(3))

# %% [4. Concatenation]
# Concatenate datasets vertically
df_concat = pd.concat([orders[:50], orders[50:]], axis=0, ignore_index=True)
print("\nConcatenated DataFrame Shape:", df_concat.shape)

# Concatenate horizontally
df_hconcat = pd.concat([orders[['order_id']], customers[['name']]], axis=1)
print("\nHorizontally Concatenated DataFrame (first 3 rows):\n", df_hconcat.head(3))

# %% [5. Resolving Merge Conflicts]
# Introduce duplicates
orders_dupe = pd.concat([orders, orders.iloc[:10]], ignore_index=True)
df_merged_dupe = pd.merge(orders_dupe, customers, on='customer_id', how='inner')
df_merged_clean = df_merged_dupe.drop_duplicates(subset=['order_id', 'customer_id'])
print("\nMerged DataFrame after Dropping Duplicates Shape:", df_merged_clean.shape)

# %% [6. Visualizing Merging]
# Visualize merge results
plt.figure(figsize=(8, 4))
plt.hist(df_merged['amount'], bins=20, color='blue', alpha=0.7, label='Merged (Inner)')
plt.hist(df_left['amount'], bins=20, color='red', alpha=0.5, label='Left Join')
plt.title('Order Amount Distribution by Join Type')
plt.xlabel('Amount ($)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('merging_joining_histogram.png')
plt.close()

# %% [7. Practical ML Application]
# Merge ML features and labels.
np.random.seed(42)
features = pd.DataFrame({
    'sample_id': range(100),
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100)
})
labels = pd.DataFrame({
    'sample_id': range(90, 110),
    'target': np.random.randint(0, 2, 20)
})
ml_df = pd.merge(features, labels, on='sample_id', how='left')
ml_df['target'] = ml_df['target'].fillna(0)  # Fill missing labels
print("\nML Merged DataFrame (first 3 rows):\n", ml_df.head(3))

# %% [8. Interview Scenario: Join Types]
# Discuss join types for ML.
print("\nInterview Scenario: Join Types")
print("Q: What’s the difference between inner and left joins in Pandas?")
print("A: Inner join keeps only matching rows; left join keeps all left rows, filling non-matches with NaN.")
print("Key: Choose join type based on data completeness needs.")
print("Example: pd.merge(df1, df2, on='key', how='left') for all left rows.")