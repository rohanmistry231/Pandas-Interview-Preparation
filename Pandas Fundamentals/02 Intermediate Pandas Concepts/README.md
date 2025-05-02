# 🏋️ Intermediate Pandas Concepts (`pandas`)

## 📖 Introduction
Pandas is a cornerstone for AI and machine learning (ML) data preprocessing, enabling advanced manipulation of structured data. This section deepens your Pandas skills with **Advanced Data Cleaning**, **Merging and Joining**, **Group-by and Aggregation**, and **Feature Engineering for ML**, building on beginner concepts (e.g., DataFrame creation, basic cleaning). With practical examples and interview insights, it prepares you for robust ML data pipelines.

## 🎯 Learning Objectives
- Clean complex datasets with outlier handling, string operations, and categorical encoding.
- Combine datasets using merging, joining, and concatenation techniques.
- Perform group-by operations and aggregations for data summarization.
- Engineer features to enhance ML model performance.

## 🔑 Key Concepts
- **Advanced Data Cleaning**:
  - Handle outliers (`quantile`, `clip`).
  - Perform string operations (`str.replace`, `str.extract`).
  - Encode categorical variables (`get_dummies`, `map`).
- **Merging and Joining**:
  - Combine datasets (`merge`, `join`, `concat`).
  - Use join types (inner, left, outer).
  - Resolve merge conflicts and duplicates.
- **Group-by and Aggregation**:
  - Group data (`groupby`, `agg`, `pivot_table`).
  - Apply custom aggregation functions.
  - Handle multi-level indexing.
- **Feature Engineering for ML**:
  - Create features (ratios, bins).
  - Normalize/standardize features.
  - Process datetime data (`to_datetime`, `dt` accessor).

## 📝 Example Walkthroughs
The following Python files demonstrate each subsection:

1. **`advanced_data_cleaning.py`**:
   - Clips outliers in Iris data using quantiles (`clip`).
   - Extracts product types and cleans strings in a retail dataset (`str.extract`, `str.replace`).
   - Encodes categories with `get_dummies` and `map`.
   - Visualizes outlier clipping (box plot).

   Example code:
   ```python
   import pandas as pd
   q_low, q_high = df['col'].quantile([0.05, 0.95])
   df['col'] = df['col'].clip(lower=q_low, upper=q_high)
   df_encoded = pd.get_dummies(df, columns=['category'])
   ```

2. **`merging_joining.py`**:
   - Merges retail orders and customer data (`merge`) with different join types.
   - Joins indexed DataFrames (`join`) and concatenates (`concat`).
   - Resolves duplicates in merges (`drop_duplicates`).
   - Visualizes join type impacts (histogram).

   Example code:
   ```python
   import pandas as pd
   df_merged = pd.merge(orders, customers, on='customer_id', how='inner')
   df_concat = pd.concat([df1, df2], axis=0)
   ```

3. **`groupby_aggregation.py`**:
   - Groups retail data by store/product (`groupby`) and aggregates (`agg`).
   - Creates pivot tables (`pivot_table`) for summarization.
   - Handles multi-level indexing (`set_index`).
   - Visualizes pivot table (heatmap).

   Example code:
   ```python
   import pandas as pd
   df_grouped = df.groupby('store').agg({'sales': ['mean', 'sum']})
   pivot = df.pivot_table(values='sales', index='store', columns='product')
   ```

4. **`feature_engineering_ml.py`**:
   - Creates ratio and binned features (`cut`).
   - Standardizes/normalizes features for ML.
   - Extracts datetime attributes (`dt.month`, `dt.day_name`).
   - Visualizes original vs. standardized features (histogram).

   Example code:
   ```python
   import pandas as pd
   df['ratio'] = df['col1'] / df['col2']
   df['std'] = (df['col'] - df['col'].mean()) / df['col'].std()
   df['month'] = pd.to_datetime(df['date']).dt.month
   ```

## 🛠️ Practical Tasks
1. **Data Cleaning**:
   - Clip outliers in a numeric column using quantiles.
   - Extract patterns from a text column with `str.extract`.
   - Encode a categorical column with one-hot encoding.
2. **Merging**:
   - Merge two datasets with a left join and handle missing values.
   - Concatenate two DataFrames vertically and remove duplicates.
3. **Group-by**:
   - Compute mean and standard deviation by group (`groupby`, `agg`).
   - Create a pivot table summarizing sales by category and region.
4. **Feature Engineering**:
   - Create a feature as the ratio of two columns.
   - Standardize a numeric feature for ML.
   - Extract the day of the week from a datetime column.

## 💡 Interview Tips
- **Common Questions**:
  - How do you handle outliers in a Pandas DataFrame?
  - What’s the difference between `merge` and `concat`?
  - How would you compute group-wise statistics for ML?
  - Why standardize features before ML training?
- **Tips**:
  - Explain outlier clipping with quantiles for robust data.
  - Highlight `merge` for key-based joins vs. `concat` for stacking.
  - Be ready to code feature engineering (e.g., `df['new'] = df['col1'] * df['col2']`).
- **Coding Tasks**:
  - Clean a dataset by clipping outliers and encoding categoricals.
  - Merge two DataFrames and resolve duplicates.
  - Create a pivot table for group-wise analysis.

## 📚 Resources
- [Pandas Merging](https://pandas.pydata.org/docs/user_guide/merging.html)
- [Pandas GroupBy](https://pandas.pydata.org/docs/user_guide/groupby.html)
- [Pandas Categorical Data](https://pandas.pydata.org/docs/user_guide/categorical.html)
- [Pandas Time Series](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [Kaggle: Pandas Course](https://www.kaggle.com/learn/pandas)
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781491957653/)