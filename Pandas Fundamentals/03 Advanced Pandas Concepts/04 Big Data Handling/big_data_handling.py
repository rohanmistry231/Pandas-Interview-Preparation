import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import dask.dataframe as dd
    from multiprocessing import Pool
except ImportError:
    dd, Pool = None, None

# %% [1. Introduction to Big Data Handling]
# Learn Pandas techniques for handling large ML datasets.
# Covers Dask, sparse DataFrames, and parallel processing.

print("Pandas version:", pd.__version__)

# %% [2. Working with Dask]
# Create large synthetic dataset.
np.random.seed(42)
with open('big_data.csv', 'w') as f:
    f.write('feature1,feature2,category\n')
    for _ in range(100000):
        f.write(f"{np.random.rand()},{np.random.rand()},{np.random.choice(['A', 'B', 'C'])}\n")

# Use Dask for out-of-memory processing
if dd:
    ddf = dd.read_csv('big_data.csv')
    mean_feature1 = ddf['feature1'].mean().compute()
    print("\nDask DataFrame Mean Feature1:", mean_feature1)
else:
    print("\nDask not available; skipping.")

# %% [3. Sparse DataFrames]
# Create sparse dataset.
np.random.seed(42)
sparse_data = np.random.rand(1000, 1000)
sparse_data[sparse_data < 0.9] = 0  # 90% sparsity
df_sparse = pd.DataFrame(sparse_data)
sparse_df = pd.DataFrame.sparse.from_spmatrix(pd.DataFrame(sparse_data).sparse.to_coo())
print("\nSparse DataFrame Memory Usage (MB):", sparse_df.memory_usage().sum() / 1024**2)
print("Dense DataFrame Memory Usage (MB):", df_sparse.memory_usage().sum() / 1024**2)

# %% [4. Parallel Processing with Multiprocessing]
# Parallelize a computation.
def compute_mean(chunk):
    return chunk['feature1'].mean()

if Pool:
    chunks = pd.read_csv('big_data.csv', chunksize=10000)
    with Pool(4) as pool:
        means = pool.map(compute_mean, [chunk for chunk in chunks])
    print("\nParallel Means (first 3):", means[:3])
else:
    print("\nMultiprocessing not available; skipping.")

# %% [5. Visualizing Big Data]
# Visualize sparse matrix structure.
plt.figure(figsize=(6, 4))
plt.spy(sparse_df.sparse.to_coo(), markersize=1)
plt.title('Sparse DataFrame Structure')
plt.savefig('big_data_sparse.png')
plt.close()

# %% [6. Practical ML Application]
# Process large ML dataset with Dask.
if dd:
    ddf_ml = dd.read_csv('big_data.csv')
    ddf_ml['feature_product'] = ddf_ml['feature1'] * ddf_ml['feature2']
    ddf_ml = ddf_ml.categorize('category')
    ddf_ml = ddf_ml.persist()
    print("\nDask ML DataFrame Head (computed):\n", ddf_ml.head(3))
else:
    np.random.seed(42)
    ml_df = pd.DataFrame({
        'feature1': np.random.rand(1000),
        'feature2': np.random.rand(1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    ml_df['feature_product'] = ml_df['feature1'] * ml_df['feature2']
    print("\nFallback ML DataFrame (first 3 rows):\n", ml_df.head(3))

# %% [7. Interview Scenario: Big Data]
# Discuss big data handling for ML.
print("\nInterview Scenario: Big Data")
print("Q: How would you handle a large dataset in Pandas for ML?")
print("A: Use Dask for out-of-memory processing or chunking with read_csv(chunksize).")
print("Key: Dask scales Pandas operations; chunking manages memory.")
print("Example: ddf = dd.read_csv('data.csv'); mean = ddf['col'].mean().compute()")