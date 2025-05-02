import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Basic Operations]
# Learn how to index, select, filter, sort, and rank data in Pandas DataFrames.
# Essential for preparing ML datasets.

print("Pandas version:", pd.__version__)

# %% [2. Indexing and Selecting Data]
# Load Iris dataset or use synthetic data.
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

# Select columns
sepal_cols = df[['sepal length (cm)', 'sepal width (cm)']]
print("\nSelected Columns (first 3 rows):\n", sepal_cols.head(3))

# loc and iloc
print("\nRow 0 with loc:\n", df.loc[0])
print("\nRow 0 with iloc:\n", df.iloc[0])

# Select specific rows and columns
subset = df.loc[:2, ['sepal length (cm)', 'petal length (cm)']]
print("\nSubset with loc (first 3 rows):\n", subset)

# %% [3. Filtering Rows]
# Boolean indexing
long_sepal = df[df['sepal length (cm)'] > 6.0]
print("\nRows with Sepal Length > 6.0 (first 3):\n", long_sepal.head(3))

# Query method
wide_sepal = df.query('`sepal width (cm)` > 3.5')
print("\nRows with Sepal Width > 3.5 (first 3):\n", wide_sepal.head(3))

# %% [4. Sorting and Ranking]
# Sort by sepal length
sorted_df = df.sort_values('sepal length (cm)', ascending=True)
print("\nSorted by Sepal Length (first 3 rows):\n", sorted_df.head(3))

# Rank petal length
df['petal_length_rank'] = df['petal length (cm)'].rank()
print("\nPetal Length Rank (first 3 rows):\n", df[['petal length (cm)', 'petal_length_rank']].head(3))

# %% [5. Visualizing Operations]
# Visualize filtered data.
plt.figure(figsize=(8, 4))
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], c='blue', alpha=0.5, label='All')
plt.scatter(long_sepal['sepal length (cm)'], long_sepal['petal length (cm)'], c='red', label='Sepal > 6.0')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.title('Filtered Data: Sepal Length > 6.0')
plt.legend()
plt.savefig('basic_operations_scatter.png')

# %% [6. Practical ML Application]
# Filter and select features for ML.
np.random.seed(42)
ml_df = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
})
filtered_ml = ml_df[ml_df['feature1'] > 0.5][['feature1', 'target']]
print("\nML Filtered DataFrame (first 3 rows):\n", filtered_ml.head(3))

# %% [7. Interview Scenario: Filtering Data]
# Discuss filtering for ML preprocessing.
print("\nInterview Scenario: Filtering Data")
print("Q: How would you filter a DataFrame for ML preprocessing?")
print("A: Use boolean indexing (df[df['col'] > value]) or df.query for conditions.")
print("Key: Filtering ensures relevant data for ML models.")
print("Example: df[df['feature'] > df['feature'].mean()] for above-average values.")