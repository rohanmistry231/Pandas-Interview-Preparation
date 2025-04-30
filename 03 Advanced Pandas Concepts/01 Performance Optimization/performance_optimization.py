import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
try:
    from numba import jit
except ImportError:
    jit = lambda x: x

# %% [1. Introduction to Performance Optimization]
# Learn advanced Pandas techniques for optimizing ML data processing.
# Covers vectorized operations, efficient storage, and numba/pandas.eval.

print("Pandas version:", pd.__version__)

# %% [2. Vectorized Operations Over Loops]
# Compare loop vs. vectorized operations.
np.random.seed(42)
df = pd.DataFrame({
    'value': np.random.rand(100000),
    'category': np.random.choice(['A', 'B', 'C'], 100000)
})

# Loop-based computation
start_time = time.time()
df['value_squared_loop'] = [x**2 for x in df['value']]
loop_time = time.time() - start_time

# Vectorized computation
start_time = time.time()
df['value_squared_vector'] = df['value'] ** 2
vector_time = time.time() - start_time
print("\nLoop Time:", loop_time, "seconds")
print("Vectorized Time:", vector_time, "seconds")
print("Speedup:", loop_time / vector_time)

# Verify results
print("Results Match:", (df['value_squared_loop'] == df['value_squared_vector']).all())

# %% [3. Efficient Data Storage]
# Save DataFrame to pickle and parquet.
df.to_pickle('data.pkl')
df.to_parquet('data.parquet')
print("\nSaved DataFrame to pickle and parquet")

# Compare file sizes
import os
print("Pickle File Size:", os.path.getsize('data.pkl') / 1024, "KB")
print("Parquet File Size:", os.path.getsize('data.parquet') / 1024, "KB")

# Read files
df_pickle = pd.read_pickle('data.pkl')
df_parquet = pd.read_parquet('data.parquet')
print("\nRead DataFrame Shapes:", df_pickle.shape, df_parquet.shape)

# %% [4. Using Numba for Speed]
# Optimize a computation with numba.
@jit(nopython=True)
def compute_sum(values):
    total = 0
    for v in values:
        total += v
    return total

start_time = time.time()
numba_sum = compute_sum(df['value'].values)
numba_time = time.time() - start_time
print("\nNumba Sum Time:", numba_time, "seconds")
print("Numba Sum Result:", numba_sum)

# %% [5. Using pandas.eval]
# Optimize DataFrame computation with eval.
start_time = time.time()
df['product'] = pd.eval('df.value * df.value_squared_vector')
eval_time = time.time() - start_time
print("\npandas.eval Time:", eval_time, "seconds")
print("Product Column (first 3 rows):\n", df['product'].head(3))

# %% [6. Visualizing Performance]
# Plot loop vs. vectorized times.
plt.figure(figsize=(8, 4))
plt.bar(['Loop', 'Vectorized', 'Numba', 'Eval'], [loop_time, vector_time, numba_time, eval_time], color=['red', 'green', 'blue', 'purple'])
plt.title('Performance Comparison')
plt.ylabel('Time (seconds)')
plt.savefig('performance_optimization_bar.png')
plt.close()

# %% [7. Practical ML Application]
# Optimize feature computation for ML.
np.random.seed(42)
ml_df = pd.DataFrame({
    'feature1': np.random.rand(100000),
    'feature2': np.random.rand(100000)
})
start_time = time.time()
ml_df['feature_product'] = pd.eval('ml_df.feature1 * ml_df.feature2')
ml_df.to_parquet('ml_features.parquet')
opt_time = time.time() - start_time
print("\nML Feature Optimization Time:", opt_time, "seconds")
print("Optimized ML DataFrame (first 3 rows):\n", ml_df.head(3))

# %% [8. Interview Scenario: Optimization]
# Discuss performance optimization for ML.
print("\nInterview Scenario: Optimization")
print("Q: How would you optimize a slow Pandas operation for ML?")
print("A: Use vectorized operations, pandas.eval, or numba; save to parquet for efficiency.")
print("Key: Vectorization leverages C-based operations; parquet reduces storage.")
print("Example: df['new'] = df['col1'] * df['col2'] vs. loop.")