import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.datasets import load_iris
except ImportError:
    load_iris = None

# %% [1. Introduction to Data Visualization]
# Learn how to visualize Pandas DataFrames for ML insights.
# Covers plotting with Pandas (hist, box) and Matplotlib customization.

print("Pandas version:", pd.__version__)

# %% [2. Plotting with Pandas]
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

# Histogram
df['sepal length (cm)'].plot.hist(bins=20, color='blue', alpha=0.7, title='Sepal Length Histogram')
plt.xlabel('Sepal Length (cm)')
plt.savefig('data_visualization_histogram.png')
plt.close()

# Box plot
df.plot.box(title='Feature Box Plots')
plt.ylabel('Value (cm)')
plt.savefig('data_visualization_box.png')
plt.close()

# %% [3. Customizing Visualizations with Matplotlib]
# Scatter plot with Matplotlib
plt.figure(figsize=(8, 4))
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], c='purple', alpha=0.5)
plt.title('Sepal vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.grid(True)
plt.savefig('data_visualization_scatter.png')
plt.close()

# Combined histograms
plt.figure(figsize=(8, 4))
for column in df.columns:
    df[column].hist(bins=15, alpha=0.5, label=column)
plt.title('Feature Distributions')
plt.xlabel('Value (cm)')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('data_visualization_combined_hist.png')
plt.close()

# %% [4. Visualizing ML Insights]
# Visualize feature correlations.
corr_matrix = df.corr()
plt.figure(figsize=(6, 4))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
plt.title('Feature Correlation Matrix')
plt.savefig('data_visualization_correlation.png')
plt.close()

# %% [5. Practical ML Application]
# Visualize a synthetic ML dataset.
np.random.seed(42)
ml_df = pd.DataFrame({
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
})
ml_df[ml_df['target'] == 0][['feature1', 'feature2']].plot.scatter(
    x='feature1', y='feature2', c='blue', label='Class 0', alpha=0.5
)
ml_df[ml_df['target'] == 1][['feature1', 'feature2']].plot.scatter(
    x='feature1', y='feature2', c='red', label='Class 1', alpha=0.5
)
plt.title('ML Dataset: Feature Scatter')
plt.savefig('data_visualization_ml_scatter.png')
plt.close()

# %% [6. Interview Scenario: Visualization]
# Discuss visualization for ML insights.
print("\nInterview Scenario: Visualization")
print("Q: How would you visualize a dataset’s features for ML?")
print("A: Use df.plot.hist for distributions, df.plot.scatter for relationships.")
print("Key: Visualizations reveal patterns and outliers.")
print("Example: df.plot.box() to identify outliers in features.")