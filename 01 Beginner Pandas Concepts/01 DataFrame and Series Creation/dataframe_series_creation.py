import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to DataFrame and Series Creation]
# Learn how to create and explore Pandas DataFrames and Series for ML data handling.
# Covers pd.DataFrame, pd.Series, pd.read_csv, and data exploration methods.

print("Pandas version:", pd.__version__)

# %% [2. Creating DataFrames]
# Create DataFrame from a dictionary or Iris dataset.
if load_iris:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
else:
    # Fallback synthetic data
    data = {
        'sepal length (cm)': [5.1, 4.9, 7.0],
        'sepal width (cm)': [3.5, 3.0, 3.2],
        'petal length (cm)': [1.4, 1.4, 4.7],
        'petal width (cm)': [0.2, 0.2, 1.4]
    }
    df = pd.DataFrame(data)

print("\nDataFrame (first 3 rows):\n", df.head(3))

# Create DataFrame from NumPy array
np.random.seed(42)
np_array = np.random.rand(5, 3)
df_np = pd.DataFrame(np_array, columns=['Feature1', 'Feature2', 'Feature3'])
print("\nDataFrame from NumPy (first 3 rows):\n", df_np.head(3))

# %% [3. Creating Series]
# Create Series from list or DataFrame column.
series = pd.Series([5.1, 4.9, 7.0], name='sepal_length')
print("\nSeries:\n", series.head())

# Extract Series from DataFrame
sepal_length_series = df['sepal length (cm)']
print("\nSeries from DataFrame (first 3):\n", sepal_length_series.head(3))

# %% [4. Importing/Exporting Data]
# Simulate CSV import/export with Iris data.
df.to_csv('iris_data.csv', index=False)
df_csv = pd.read_csv('iris_data.csv')
print("\nImported CSV DataFrame (first 3 rows):\n", df_csv.head(3))

# JSON import/export
df.to_json('iris_data.json')
df_json = pd.read_json('iris_data.json')
print("\nImported JSON DataFrame (first 3 rows):\n", df_json.head(3))

# %% [5. Exploring Data]
# Use head, info, describe for data exploration.
print("\nDataFrame Info:")
df.info()

print("\nDataFrame Description:\n", df.describe())

print("\nFirst 5 Rows:\n", df.head())

# %% [6. Visualizing Data Creation]
# Visualize feature distribution.
plt.figure(figsize=(8, 4))
df['sepal length (cm)'].hist(bins=20, color='blue', alpha=0.7)
plt.title('Sepal Length Distribution')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.savefig('dataframe_series_histogram.png')

# %% [7. Practical ML Application]
# Create a synthetic ML dataset with DataFrame.
np.random.seed(42)
ml_data = {
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
}
df_ml = pd.DataFrame(ml_data)
print("\nSynthetic ML DataFrame (first 3 rows):\n", df_ml.head(3))
df_ml.to_csv('ml_dataset.csv', index=False)

# %% [8. Interview Scenario: DataFrame Creation]
# Discuss creating DataFrames for ML.
print("\nInterview Scenario: DataFrame Creation")
print("Q: How would you create a DataFrame for an ML dataset?")
print("A: Use pd.DataFrame from a dictionary or pd.read_csv for external data.")
print("Key: Ensure correct column names and data types for ML compatibility.")
print("Example: df = pd.DataFrame({'feature': data, 'target': labels})")