import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% [1. Introduction to Group-by and Aggregation]
# Learn Pandas group-by and aggregation for ML data analysis.
# Covers groupby, agg, pivot_table, and multi-level indexing.

print("Pandas version:", pd.__version__)

# %% [2. Grouping Data]
# Create synthetic retail dataset.
np.random.seed(42)
retail_data = {
    'store': np.random.choice(['A', 'B', 'C'], 200),
    'product': np.random.choice(['Laptop', 'Phone', 'Tablet'], 200),
    'sales': np.random.normal(1000, 200, 200)
}
df = pd.DataFrame(retail_data)

# Group by store
store_groups = df.groupby('store')
print("\nGroup-by Store Mean Sales:\n", store_groups['sales'].mean())

# Group by store and product
multi_groups = df.groupby(['store', 'product'])
print("\nGroup-by Store and Product Sum Sales (first 3):\n", multi_groups['sales'].sum().head(3))

# %% [3. Aggregation]
# Apply multiple aggregations
agg_result = df.groupby('store').agg({'sales': ['mean', 'sum', 'count']})
print("\nAggregated Sales by Store:\n", agg_result)

# Custom aggregation function
def sales_range(x):
    return x.max() - x.min()
custom_agg = df.groupby('product').agg({'sales': sales_range})
print("\nCustom Aggregation (Sales Range by Product):\n", custom_agg)

# %% [4. Pivot Tables]
# Create pivot table
pivot = df.pivot_table(values='sales', index='store', columns='product', aggfunc='mean')
print("\nPivot Table (Mean Sales):\n", pivot)

# %% [5. Multi-level Indexing]
# Create multi-level index
multi_index_df = df.set_index(['store', 'product'])
print("\nMulti-level Index DataFrame (first 3 rows):\n", multi_index_df.head(3))

# Aggregate with multi-level index
multi_agg = multi_index_df.groupby(level=['store', 'product']).sum()
print("\nMulti-level Aggregation:\n", multi_agg.head(3))

# %% [6. Visualizing Aggregation]
# Visualize pivot table
plt.figure(figsize=(8, 4))
plt.imshow(pivot, cmap='viridis')
plt.colorbar(label='Mean Sales ($)')
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.title('Pivot Table: Mean Sales by Store and Product')
plt.savefig('groupby_aggregation_pivot.png')
plt.close()

# %% [7. Practical ML Application]
# Aggregate features for ML.
np.random.seed(42)
ml_data = {
    'sample_id': np.random.randint(1, 10, 100),
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100)
}
ml_df = pd.DataFrame(ml_data)
ml_agg = ml_df.groupby('sample_id').agg({'feature1': 'mean', 'feature2': 'std'})
print("\nAggregated ML Features (first 3 rows):\n", ml_agg.head(3))

# %% [8. Interview Scenario: Group-by]
# Discuss group-by for ML.
print("\nInterview Scenario: Group-by")
print("Q: How would you compute group-wise statistics for ML features?")
print("A: Use df.groupby('col').agg({'feature': 'mean'}) for group-wise metrics.")
print("Key: Group-by summarizes data for feature engineering.")
print("Example: df.groupby('category').agg({'value': ['mean', 'std']})")