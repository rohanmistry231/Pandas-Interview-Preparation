import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import tensorflow as tf
    import torch
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
except ImportError:
    tf, torch, make_pipeline, StandardScaler, LogisticRegression = None, None, None, None, None

# %% [1. Introduction to Integration with ML Frameworks]
# Learn how to integrate Pandas with ML frameworks for data pipelines.
# Covers conversions to NumPy/TensorFlow/PyTorch, scikit-learn pipelines, and chunking.

print("Pandas version:", pd.__version__)

# %% [2. Converting DataFrames to ML Formats]
# Create synthetic ML dataset.
np.random.seed(42)
df = pd.DataFrame({
    'feature1': np.random.rand(1000),
    'feature2': np.random.rand(1000),
    'target': np.random.randint(0, 2, 1000)
})

# Convert to NumPy
X_np = df[['feature1', 'feature2']].to_numpy()
y_np = df['target'].to_numpy()
print("\nNumPy Arrays Shapes:", X_np.shape, y_np.shape)

# Convert to TensorFlow
if tf:
    X_tf = tf.convert_to_tensor(X_np, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y_np, dtype=tf.int32)
    print("TensorFlow Tensors Shapes:", X_tf.shape, y_tf.shape)
else:
    print("TensorFlow not available; skipping.")

# Convert to PyTorch
if torch:
    X_torch = torch.from_numpy(X_np).float()
    y_torch = torch.from_numpy(y_np).long()
    print("PyTorch Tensors Shapes:", X_torch.shape, y_torch.shape)
else:
    print("PyTorch not available; skipping.")

# %% [3. Building ML Pipelines with scikit-learn]
# Create a scikit-learn pipeline.
if make_pipeline:
    pipeline = make_pipeline(StandardScaler(), LogisticRegression())
    pipeline.fit(X_np, y_np)
    score = pipeline.score(X_np, y_np)
    print("\nScikit-learn Pipeline Accuracy:", score)
else:
    print("\nScikit-learn not available; skipping pipeline.")

# %% [4. Handling Large Datasets with Chunking]
# Simulate large dataset with chunks.
np.random.seed(42)
with open('large_data.csv', 'w') as f:
    f.write('feature1,feature2,target\n')
    for _ in range(10000):
        f.write(f"{np.random.rand()},{np.random.rand()},{np.random.randint(0, 2)}\n")

# Process in chunks
chunk_size = 1000
chunks = pd.read_csv('large_data.csv', chunksize=chunk_size)
means = []
for chunk in chunks:
    means.append(chunk['feature1'].mean())
print("\nChunked Feature1 Means (first 3):", means[:3])

# %% [5. Visualizing Integration]
# Visualize feature distributions.
plt.figure(figsize=(8, 4))
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap='viridis', alpha=0.5)
plt.title('ML Features: Class Scatter')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.colorbar(label='Target')
plt.savefig('integration_ml_scatter.png')
plt.close()

# %% [6. Practical ML Application]
# Prepare DataFrame for a deep learning pipeline.
np.random.seed(42)
ml_df = pd.DataFrame({
    'feature1': np.random.rand(1000),
    'feature2': np.random.rand(1000),
    'target': np.random.randint(0, 2, 1000)
})
if tf:
    dataset = tf.data.Dataset.from_tensor_slices(
        (ml_df[['feature1', 'feature2']].to_numpy(), ml_df['target'].to_numpy())
    ).batch(32)
    print("\nTensorFlow Dataset Batch Shapes:")
    for X_batch, y_batch in dataset.take(1):
        print(X_batch.shape, y_batch.shape)
else:
    print("\nTensorFlow not available; skipping dataset.")

# %% [7. Interview Scenario: Framework Integration]
# Discuss integration for ML.
print("\nInterview Scenario: Framework Integration")
print("Q: How do you prepare a Pandas DataFrame for TensorFlow?")
print("A: Convert to NumPy with df.to_numpy, then to tensors with tf.convert_to_tensor.")
print("Key: Use tf.data.Dataset for efficient batching.")
print("Example: dataset = tf.data.Dataset.from_tensor_slices((X_np, y_np)).batch(32)")