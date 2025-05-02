# Pandas Interview Questions for AI/ML Roles

This README provides 170 Pandas interview questions tailored for AI/ML roles, focusing on data manipulation and analysis with Pandas in Python. The questions cover **core Pandas concepts** (e.g., DataFrame creation, indexing, merging, grouping, cleaning) and their applications in AI/ML tasks like data preprocessing, feature engineering, and exploratory data analysis. Questions are categorized by topic and divided into **Basic**, **Intermediate**, and **Advanced** levels to support candidates preparing for roles requiring Pandas in AI/ML workflows.

## DataFrame Creation and Manipulation

### Basic
1. **What is Pandas, and why is it important in AI/ML?**  
   Pandas provides efficient data structures for data manipulation in AI/ML.  
   ```python
   import pandas as pd
   df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
   ```

2. **How do you create a Pandas DataFrame from a dictionary?**  
   Converts dictionaries to DataFrames for analysis.  
   ```python
   data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
   df = pd.DataFrame(data)
   ```

3. **How do you create a DataFrame from a CSV file?**  
   Loads external data for processing.  
   ```python
   df = pd.read_csv('data.csv')
   ```

4. **How do you create a DataFrame with random values?**  
   Generates synthetic data for testing.  
   ```python
   import numpy as np
   df = pd.DataFrame(np.random.rand(3, 2), columns=['A', 'B'])
   ```

5. **How do you view the first few rows of a DataFrame?**  
   Inspects data structure.  
   ```python
   df.head()
   ```

6. **How do you check the basic information of a DataFrame?**  
   Summarizes columns and dtypes.  
   ```python
   df.info()
   ```

#### Intermediate
7. **Write a function to create a DataFrame from a list of dictionaries.**  
   Structures dynamic data.  
   ```python
   def create_df_from_dicts(dict_list):
       return pd.DataFrame(dict_list)
   ```

8. **How do you create a DataFrame with a MultiIndex?**  
   Supports hierarchical indexing.  
   ```python
   arrays = [['A', 'A', 'B', 'B'], [1, 2, 1, 2]]
   df = pd.DataFrame(np.random.rand(4, 2), index=pd.MultiIndex.from_arrays(arrays))
   ```

9. **Write a function to initialize a DataFrame with missing values.**  
   Simulates real-world data.  
   ```python
   def create_df_with_nans(rows, cols):
       data = np.random.rand(rows, cols)
       data[np.random.rand(rows, cols) < 0.2] = np.nan
       return pd.DataFrame(data, columns=[f'Col{i}' for i in range(cols)])
   ```

10. **How do you convert a DataFrame to a NumPy array?**  
    Prepares data for ML models.  
    ```python
    array = df.to_numpy()
    ```

11. **Write a function to visualize a DataFrame column distribution.**  
    Plots histograms for analysis.  
    ```python
    import matplotlib.pyplot as plt
    def plot_column_dist(df, column):
        df[column].hist(bins=20)
        plt.savefig('column_dist.png')
    ```

12. **How do you rename columns in a DataFrame?**  
    Updates column names for clarity.  
    ```python
    df.rename(columns={'A': 'X', 'B': 'Y'}, inplace=True)
    ```

#### Advanced
13. **Write a function to create a DataFrame with custom dtypes.**  
    Optimizes memory usage.  
    ```python
    def create_typed_df(data, dtypes):
        return pd.DataFrame(data).astype(dtypes)
    ```

14. **How do you optimize DataFrame creation for large datasets?**  
    Uses chunked reading.  
    ```python
    chunks = pd.read_csv('large_data.csv', chunksize=1000)
    df = pd.concat(chunks)
    ```

15. **Write a function to create a pivoted DataFrame.**  
    Reshapes data for analysis.  
    ```python
    def pivot_df(df, index, columns, values):
        return df.pivot(index=index, columns=columns, values=values)
    ```

16. **How do you handle memory-efficient DataFrame creation?**  
    Uses sparse formats or categorical types.  
    ```python
    df['category'] = df['category'].astype('category')
    ```

17. **Write a function to create a DataFrame from a SQL query.**  
    Integrates with databases.  
    ```python
    import sqlite3
    def sql_to_df(query, db_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    ```

18. **How do you create a DataFrame with time series data?**  
    Structures temporal data.  
    ```python
    dates = pd.date_range('2023-01-01', periods=5)
    df = pd.DataFrame({'value': np.random.rand(5)}, index=dates)
    ```

## Indexing and Selection

### Basic
19. **How do you select a column in a Pandas DataFrame?**  
   Accesses specific features.  
   ```python
   column = df['A']
   ```

20. **What is the difference between `loc` and `iloc` in Pandas?**  
   `loc` uses labels; `iloc` uses indices.  
   ```python
   df.loc[0, 'A']
   df.iloc[0, 0]
   ```

21. **How do you filter rows in a DataFrame based on a condition?**  
   Selects data dynamically.  
   ```python
   filtered = df[df['A'] > 2]
   ```

22. **How do you select multiple columns in a DataFrame?**  
   Extracts feature subsets.  
   ```python
   subset = df[['A', 'B']]
   ```

23. **How do you access a specific cell in a DataFrame?**  
   Retrieves single values.  
   ```python
   value = df.at[0, 'A']
   ```

24. **How do you visualize selected DataFrame data?**  
   Plots filtered data.  
   ```python
   import matplotlib.pyplot as plt
   df[df['A'] > 2]['B'].plot()
   plt.savefig('filtered_plot.png')
   ```

#### Intermediate
25. **Write a function to select rows based on multiple conditions.**  
    Filters with complex logic.  
    ```python
    def filter_rows(df, conditions):
        mask = np.logical_and.reduce(conditions)
        return df[mask]
    ```

26. **How do you use query to filter a DataFrame?**  
    Simplifies conditional filtering.  
    ```python
    result = df.query('A > 2 and B < 5')
    ```

27. **Write a function to select data with a MultiIndex.**  
    Accesses hierarchical data.  
    ```python
    def select_multiindex(df, level1, level2):
        return df.loc[(level1, level2)]
    ```

28. **How do you optimize DataFrame indexing for performance?**  
    Uses efficient access methods.  
    ```python
    value = df.iat[0, 0]
    ```

29. **Write a function to extract a random sample from a DataFrame.**  
    Supports data exploration.  
    ```python
    def sample_df(df, n=5):
        return df.sample(n)
    ```

30. **How do you handle missing indices in a DataFrame?**  
    Reindexes or fills gaps.  
    ```python
    df.reindex(range(10), fill_value=0)
    ```

#### Advanced
31. **Write a function to implement dynamic indexing in Pandas.**  
    Selects data based on runtime criteria.  
    ```python
    def dynamic_index(df, column, value):
        return df[df[column] == value]
    ```

32. **How do you use advanced indexing with Pandas?**  
    Combines label and position-based indexing.  
    ```python
    result = df.loc[df.index[:2], ['A', 'B']]
    ```

33. **Write a function to select data with time-based indexing.**  
    Filters temporal data.  
    ```python
    def time_index_select(df, start, end):
        return df.loc[start:end]
    ```

34. **How do you optimize indexing for large DataFrames?**  
    Uses vectorized operations.  
    ```python
    result = df.loc[df['A'].values > 2]
    ```

35. **Write a function to handle hierarchical indexing.**  
    Manages MultiIndex access.  
    ```python
    def hierarchical_select(df, levels):
        return df.xs(levels, level=[0, 1])
    ```

36. **How do you implement custom indexing for Pandas?**  
    Defines specialized access patterns.  
    ```python
    def custom_index(df, pattern='even'):
        if pattern == 'even':
            return df.iloc[::2]
        return df.iloc[1::2]
    ```

## Merging and Joining

### Basic
37. **How do you merge two DataFrames in Pandas?**  
   Combines datasets on keys.  
   ```python
   df1 = pd.DataFrame({'key': ['A', 'B'], 'value': [1, 2]})
   df2 = pd.DataFrame({'key': ['A', 'C'], 'value': [3, 4]})
   merged = pd.merge(df1, df2, on='key')
   ```

38. **What is the difference between merge and join in Pandas?**  
   Merge uses keys; join uses indices.  
   ```python
   df1.join(df2, how='inner')
   ```

39. **How do you perform a left join in Pandas?**  
   Retains all rows from the left DataFrame.  
   ```python
   merged = pd.merge(df1, df2, on='key', how='left')
   ```

40. **How do you concatenate DataFrames in Pandas?**  
   Stacks DataFrames vertically or horizontally.  
   ```python
   concatenated = pd.concat([df1, df2], axis=0)
   ```

41. **How do you handle duplicate keys during merging?**  
   Specifies merge behavior.  
   ```python
   merged = pd.merge(df1, df2, on='key', validate='one_to_one')
   ```

42. **How do you visualize merged DataFrame data?**  
   Plots combined data.  
   ```python
   import matplotlib.pyplot as plt
   merged['value_x'].plot()
   plt.savefig('merged_plot.png')
   ```

#### Intermediate
43. **Write a function to merge multiple DataFrames.**  
    Combines several datasets.  
    ```python
    def merge_dfs(dfs, key):
        return pd.concat(dfs, axis=0).groupby(key).sum().reset_index()
    ```

44. **How do you perform a merge with multiple keys?**  
    Joins on multiple columns.  
    ```python
    merged = pd.merge(df1, df2, on=['key1', 'key2'])
    ```

45. **Write a function to concatenate DataFrames with alignment.**  
    Handles mismatched indices.  
    ```python
    def align_concat(dfs):
        return pd.concat(dfs, join='outer')
    ```

46. **How do you optimize merging for large DataFrames?**  
    Uses efficient join types.  
    ```python
    merged = pd.merge(df1, df2, on='key', how='inner')
    ```

47. **Write a function to perform a fuzzy merge in Pandas.**  
    Matches similar keys.  
    ```python
    from fuzzywuzzy import process
    def fuzzy_merge(df1, df2, key, threshold=90):
        matches = [process.extractOne(k, df2[key])[0] for k in df1[key]]
        df2_matched = df2[df2[key].isin(matches)]
        return pd.merge(df1, df2_matched, on=key)
    ```

48. **How do you handle missing values during merging?**  
    Fills or drops NaNs post-merge.  
    ```python
    merged = pd.merge(df1, df2, on='key', how='left').fillna(0)
    ```

#### Advanced
49. **Write a function to implement a time-based merge.**  
    Joins on time intervals.  
    ```python
    def time_merge(df1, df2, time_col, tolerance='1H'):
        return pd.merge_asof(df1, df2, on=time_col, tolerance=pd.Timedelta(tolerance))
    ```

50. **How do you optimize merge operations for memory efficiency?**  
    Uses chunked merging.  
    ```python
    def chunked_merge(df1, df2, key, chunk_size=1000):
        chunks = [df1[i:i+chunk_size] for i in range(0, len(df1), chunk_size)]
        return pd.concat([pd.merge(chunk, df2, on=key) for chunk in chunks])
    ```

51. **Write a function to perform a many-to-many merge.**  
    Handles complex relationships.  
    ```python
    def many_to_many_merge(df1, df2, key):
        return pd.merge(df1, df2, on=key, how='outer')
    ```

52. **How do you implement a merge with custom logic?**  
    Applies specialized matching.  
    ```python
    def custom_merge(df1, df2, key, condition):
        return df1.merge(df2[df2[key].apply(condition)], on=key)
    ```

53. **Write a function to validate merge integrity.**  
    Checks for data consistency.  
    ```python
    def validate_merge(df1, df2, key):
        merged = pd.merge(df1, df2, on=key, validate='one_to_one')
        return merged
    ```

54. **How do you handle merge conflicts in Pandas?**  
    Resolves key overlaps.  
    ```python
    merged = pd.merge(df1, df2, on='key', suffixes=('_left', '_right'))
    ```

## Grouping and Aggregation

### Basic
55. **How do you group data in a Pandas DataFrame?**  
   Aggregates data by categories.  
   ```python
   grouped = df.groupby('A')
   ```

56. **How do you compute the mean of grouped data?**  
   Summarizes group statistics.  
   ```python
   means = df.groupby('A')['B'].mean()
   ```

57. **How do you apply multiple aggregations in Pandas?**  
   Computes various statistics.  
   ```python
   aggs = df.groupby('A').agg({'B': ['mean', 'sum']})
   ```

58. **How do you count rows in each group?**  
   Tallies group sizes.  
   ```python
   counts = df.groupby('A').size()
   ```

59. **How do you reset the index after grouping?**  
   Converts group keys to columns.  
   ```python
   result = df.groupby('A').mean().reset_index()
   ```

60. **How do you visualize grouped data?**  
   Plots group statistics.  
   ```python
   import matplotlib.pyplot as plt
   df.groupby('A')['B'].mean().plot(kind='bar')
   plt.savefig('grouped_plot.png')
   ```

#### Intermediate
61. **Write a function to perform custom grouping in Pandas.**  
    Groups with dynamic logic.  
    ```python
    def custom_group(df, column, agg_func):
        return df.groupby(column).agg(agg_func)
    ```

62. **How do you group by multiple columns?**  
    Aggregates hierarchically.  
    ```python
    result = df.groupby(['A', 'B']).sum()
    ```

63. **Write a function to apply multiple aggregations dynamically.**  
    Computes flexible statistics.  
    ```python
    def multi_agg(df, group_col, value_col, aggs):
        return df.groupby(group_col)[value_col].agg(aggs)
    ```

64. **How do you optimize grouping for large DataFrames?**  
    Uses efficient aggregations.  
    ```python
    result = df.groupby('A')['B'].agg('mean')
    ```

65. **Write a function to transform grouped data.**  
    Applies group-wise transformations.  
    ```python
    def group_transform(df, group_col, value_col):
        return df.groupby(group_col)[value_col].transform(lambda x: x - x.mean())
    ```

66. **How do you handle missing groups in aggregation?**  
    Fills or drops missing groups.  
    ```python
    result = df.groupby('A').sum().fillna(0)
    ```

#### Advanced
67. **Write a function to implement rolling aggregations in Pandas.**  
    Computes statistics over windows.  
    ```python
    def rolling_agg(df, column, window):
        return df[column].rolling(window).mean()
    ```

68. **How do you group by time intervals in Pandas?**  
    Aggregates temporal data.  
    ```python
    result = df.groupby(pd.Grouper(key='date', freq='D')).sum()
    ```

69. **Write a function to perform hierarchical grouping.**  
    Groups with multiple levels.  
    ```python
    def hierarchical_group(df, levels, agg_func):
        return df.groupby(levels).agg(agg_func)
    ```

70. **How do you optimize grouping for memory efficiency?**  
    Uses chunked processing.  
    ```python
    def chunked_group(df, group_col, chunk_size=1000):
        chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        return pd.concat([chunk.groupby(group_col).sum() for chunk in chunks])
    ```

71. **Write a function to apply custom aggregations.**  
    Defines specialized statistics.  
    ```python
    def custom_agg(df, group_col, value_col):
        return df.groupby(group_col)[value_col].apply(lambda x: x.max() - x.min())
    ```

72. **How do you implement parallel grouping in Pandas?**  
    Uses multiprocessing for speed.  
    ```python
    from joblib import Parallel, delayed
    def parallel_group(df, group_col, agg_func):
        groups = df.groupby(group_col)
        results = Parallel(n_jobs=-1)(delayed(agg_func)(group) for _, group in groups)
        return pd.concat(results)
    ```

## Data Cleaning and Preprocessing

### Basic
73. **How do you handle missing values in a DataFrame?**  
   Fills or drops NaNs.  
   ```python
   df.fillna(0, inplace=True)
   ```

74. **How do you drop duplicate rows in a DataFrame?**  
   Removes redundant data.  
   ```python
   df.drop_duplicates(inplace=True)
   ```

75. **How do you replace values in a DataFrame?**  
   Updates specific entries.  
   ```python
   df['A'].replace(1, 100, inplace=True)
   ```

76. **How do you detect missing values in a DataFrame?**  
   Identifies NaNs for cleaning.  
   ```python
   missing = df.isna().sum()
   ```

77. **How do you encode categorical variables in Pandas?**  
   Prepares data for ML.  
   ```python
   df['category'] = df['category'].astype('category').cat.codes
   ```

78. **How do you visualize missing data patterns?**  
   Plots NaN distributions.  
   ```python
   import matplotlib.pyplot as plt
   df.isna().sum().plot(kind='bar')
   plt.savefig('missing_data_plot.png')
   ```

#### Intermediate
79. **Write a function to impute missing values in a DataFrame.**  
    Fills NaNs with dynamic logic.  
    ```python
    def impute_missing(df, column, method='mean'):
        if method == 'mean':
            df[column].fillna(df[column].mean(), inplace=True)
        return df
    ```

80. **How do you handle outliers in a DataFrame?**  
    Filters extreme values.  
    ```python
    df = df[df['A'].between(df['A'].quantile(0.05), df['A'].quantile(0.95))]
    ```

81. **Write a function to clean text data in a DataFrame.**  
    Preprocesses text columns.  
    ```python
    def clean_text(df, column):
        df[column] = df[column].str.lower().str.strip()
        return df
    ```

82. **How do you optimize data cleaning for large DataFrames?**  
    Uses vectorized operations.  
    ```python
    df['A'] = df['A'].fillna(df['A'].mean())
    ```

83. **Write a function to normalize numerical columns.**  
    Scales features for ML.  
    ```python
    def normalize_column(df, column):
        df[column] = (df[column] - df[column].mean()) / df[column].std()
        return df
    ```

84. **How do you handle inconsistent data types in a DataFrame?**  
    Converts to consistent types.  
    ```python
    df['A'] = pd.to_numeric(df['A'], errors='coerce')
    ```

#### Advanced
85. **Write a function to implement advanced imputation.**  
    Uses interpolation or ML-based methods.  
    ```python
    def advanced_impute(df, column):
        df[column] = df[column].interpolate(method='linear')
        return df
    ```

86. **How do you handle high-cardinality categorical variables?**  
    Reduces dimensionality.  
    ```python
    top_n = df['category'].value_counts().index[:10]
    df['category'] = df['category'].where(df['category'].isin(top_n), 'other')
    ```

87. **Write a function to detect and remove outliers.**  
    Uses statistical methods.  
    ```python
    def remove_outliers(df, column):
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[~df[column].between(q1 - 1.5*iqr, q3 + 1.5*iqr)]
        return df
    ```

88. **How do you implement data validation in Pandas?**  
    Checks data integrity.  
    ```python
    def validate_data(df, column, min_val, max_val):
        if not df[column].between(min_val, max_val).all():
            raise ValueError("Data out of range")
        return df
    ```

89. **Write a function to preprocess time series data.**  
    Handles temporal cleaning.  
    ```python
    def preprocess_timeseries(df, time_col):
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
        return df
    ```

90. **How do you automate data cleaning pipelines?**  
    Scripts end-to-end cleaning.  
    ```python
    def clean_pipeline(df):
        df = df.dropna().drop_duplicates()
        df = normalize_column(df, 'A')
        return df
    ```

## Visualization and Interpretation

### Basic
91. **How do you visualize a DataFrame column?**  
   Plots distributions or trends.  
   ```python
   import matplotlib.pyplot as plt
   df['A'].plot(kind='hist')
   plt.savefig('column_hist.png')
   ```

92. **How do you create a scatter plot with Pandas?**  
   Visualizes relationships.  
   ```python
   df.plot.scatter(x='A', y='B')
   plt.savefig('scatter_plot.png')
   ```

93. **How do you plot a time series in Pandas?**  
   Visualizes temporal data.  
   ```python
   df.set_index('date')['value'].plot()
   plt.savefig('timeseries_plot.png')
   ```

94. **How do you visualize group statistics?**  
   Plots aggregated data.  
   ```python
   df.groupby('A')['B'].mean().plot(kind='bar')
   plt.savefig('group_stats.png')
   ```

95. **How do you create a box plot in Pandas?**  
   Shows data distributions.  
   ```python
   df.boxplot(column='A')
   plt.savefig('box_plot.png')
   ```

96. **How do you visualize correlations in a DataFrame?**  
   Plots correlation matrices.  
   ```python
   import matplotlib.pyplot as plt
   plt.imshow(df.corr(), cmap='coolwarm')
   plt.colorbar()
   plt.savefig('corr_plot.png')
   ```

#### Intermediate
97. **Write a function to visualize multiple columns.**  
    Plots comparative distributions.  
    ```python
    import matplotlib.pyplot as plt
    def plot_multi_columns(df, columns):
        df[columns].plot(kind='hist', alpha=0.5)
        plt.savefig('multi_hist.png')
    ```

98. **How do you create a pair plot with Pandas?**  
    Visualizes feature relationships.  
    ```python
    import seaborn as sns
    sns.pairplot(df)
    plt.savefig('pair_plot.png')
    ```

99. **Write a function to visualize time series trends.**  
    Plots temporal patterns.  
    ```python
    import matplotlib.pyplot as plt
    def plot_timeseries(df, column, date_col):
        df.set_index(date_col)[column].plot()
        plt.savefig('timeseries_trend.png')
    ```

100. **How do you visualize missing data patterns?**  
     Plots NaN distributions.  
     ```python
     import seaborn as sns
     sns.heatmap(df.isna())
     plt.savefig('missing_heatmap.png')
     ```

101. **Write a function to plot feature importance.**  
     Visualizes model weights.  
     ```python
     import matplotlib.pyplot as plt
     def plot_feature_importance(features, importances):
         pd.Series(importances, index=features).plot(kind='bar')
         plt.savefig('feature_importance.png')
     ```

102. **How do you visualize DataFrame outliers?**  
     Plots extreme values.  
     ```python
     df.boxplot(column='A')
     plt.savefig('outlier_plot.png')
     ```

#### Advanced
103. **Write a function to visualize high-dimensional data.**  
     Uses PCA for projection.  
     ```python
     from sklearn.decomposition import PCA
     import matplotlib.pyplot as plt
     def plot_high_dim(df, columns):
         pca = PCA(n_components=2)
         reduced = pca.fit_transform(df[columns])
         plt.scatter(reduced[:, 0], reduced[:, 1])
         plt.savefig('high_dim_plot.png')
     ```

104. **How do you implement a dashboard for Pandas metrics?**  
     Displays real-time stats.  
     ```python
     from fastapi import FastAPI
     app = FastAPI()
     metrics = []
     @app.get('/metrics')
     async def get_metrics():
         return {'metrics': metrics}
     ```

105. **Write a function to visualize data drift.**  
     Tracks data changes.  
     ```python
     import matplotlib.pyplot as plt
     def plot_data_drift(df1, df2, column):
         df1[column].hist(alpha=0.5, label='df1')
         df2[column].hist(alpha=0.5, label='df2')
         plt.legend()
         plt.savefig('data_drift.png')
     ```

106. **How do you visualize model performance with Pandas?**  
     Plots metrics like accuracy.  
     ```python
     import matplotlib.pyplot as plt
     def plot_model_metrics(metrics):
         pd.Series(metrics).plot()
         plt.savefig('model_metrics.png')
     ```

107. **Write a function to visualize categorical data.**  
     Plots category distributions.  
     ```python
     import matplotlib.pyplot as plt
     def plot_categorical(df, column):
         df[column].value_counts().plot(kind='bar')
         plt.savefig('categorical_plot.png')
     ```

108. **How do you visualize data fairness in Pandas?**  
     Plots group-wise metrics.  
     ```python
     import matplotlib.pyplot as plt
     def plot_fairness(df, group_col, metric_col):
         df.groupby(group_col)[metric_col].mean().plot(kind='bar')
         plt.savefig('fairness_plot.png')
     ```

## Integration with AI/ML Workflows

### Basic
109. **How do you preprocess data with Pandas for ML?**  
     Cleans and transforms features.  
     ```python
     df = df.dropna()
     X = df[['A', 'B']].to_numpy()
     y = df['target'].to_numpy()
     ```

110. **How do you create feature matrices with Pandas?**  
     Structures data for models.  
     ```python
     X = df[['feature1', 'feature2']]
     ```

111. **How do you split data into train/test sets with Pandas?**  
     Prepares data for evaluation.  
     ```python
     train = df.sample(frac=0.8)
     test = df.drop(train.index)
     ```

112. **How do you encode categorical features in Pandas?**  
     Prepares data for ML.  
     ```python
     df['category'] = pd.get_dummies(df['category'])
     ```

113. **How do you compute feature correlations with Pandas?**  
     Analyzes relationships.  
     ```python
     correlations = df.corr()
     ```

114. **How do you visualize feature distributions for ML?**  
     Plots histograms.  
     ```python
     import matplotlib.pyplot as plt
     df[['feature1', 'feature2']].hist()
     plt.savefig('feature_dist.png')
     ```

#### Intermediate
115. **Write a function to preprocess text features for ML.**  
     Cleans and vectorizes text.  
     ```python
     from sklearn.feature_extraction.text import TfidfVectorizer
     def preprocess_text(df, column):
         vectorizer = TfidfVectorizer()
         X = vectorizer.fit_transform(df[column])
         return X, vectorizer
     ```

116. **How do you handle imbalanced datasets with Pandas?**  
     Resamples data for balance.  
     ```python
     from sklearn.utils import resample
     def balance_df(df, target_col):
         majority = df[df[target_col] == 0]
         minority = df[df[target_col] == 1]
         minority_upsampled = resample(minority, n_samples=len(majority))
         return pd.concat([majority, minority_upsampled])
     ```

117. **Write a function to create lagged features in Pandas.**  
     Supports time series models.  
     ```python
     def create_lagged_features(df, column, lags):
         for lag in range(1, lags + 1):
             df[f'{column}_lag{lag}'] = df[column].shift(lag)
         return df
     ```

118. **How do you integrate Pandas with Scikit-learn?**  
     Prepares data for ML pipelines.  
     ```python
     from sklearn.linear_model import LogisticRegression
     X = df[['A', 'B']].to_numpy()
     y = df['target'].to_numpy()
     model = LogisticRegression().fit(X, y)
     ```

119. **Write a function to standardize features in Pandas.**  
     Scales features for ML.  
     ```python
     def standardize_features(df, columns):
         df[columns] = (df[columns] - df[columns].mean()) / df[columns].std()
         return df
     ```

120. **How do you handle feature selection with Pandas?**  
     Selects relevant features.  
     ```python
     def select_features(df, target_col, threshold=0.1):
         correlations = df.corr()[target_col].abs()
         return df[correlations[correlations > threshold].index]
     ```

#### Advanced
121. **Write a function to implement PCA with Pandas.**  
     Reduces dimensionality.  
     ```python
     from sklearn.decomposition import PCA
     def pca_transform(df, columns, n_components):
         pca = PCA(n_components=n_components)
         reduced = pca.fit_transform(df[columns])
         return pd.DataFrame(reduced, columns=[f'PC{i}' for i in range(n_components)])
     ```

122. **How do you optimize Pandas for large-scale ML datasets?**  
     Uses chunked processing.  
     ```python
     def process_chunks(df, chunk_size=1000):
         chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
         return pd.concat([standardize_features(chunk, ['A', 'B']) for chunk in chunks])
     ```

123. **Write a function to create interaction features.**  
     Enhances model performance.  
     ```python
     def interaction_features(df, col1, col2):
         df[f'{col1}_{col2}'] = df[col1] * df[col2]
         return df
     ```

124. **How do you implement automated feature engineering?**  
     Generates features dynamically.  
     ```python
     def auto_feature_engineer(df, columns):
         for col in columns:
             df[f'{col}_squared'] = df[col] ** 2
         return df
     ```

125. **Write a function to handle time series feature engineering.**  
     Creates temporal features.  
     ```python
     def timeseries_features(df, time_col):
         df['year'] = df[time_col].dt.year
         df['month'] = df[time_col].dt.month
         return df
     ```

126. **How do you integrate Pandas with deep learning frameworks?**  
     Converts data for TensorFlow/PyTorch.  
     ```python
     import tensorflow as tf
     X = df[['A', 'B']].to_numpy()
     tensor = tf.convert_to_tensor(X)
     ```

## Debugging and Error Handling

### Basic
127. **How do you debug Pandas DataFrame operations?**  
     Logs intermediate outputs.  
     ```python
     def debug_df(df):
         print(f"Shape: {df.shape}, Columns: {df.columns}")
         return df
     ```

128. **What is a try-except block in Pandas applications?**  
     Handles data errors.  
     ```python
     try:
         df['A'] = pd.to_numeric(df['A'])
     except ValueError as e:
         print(f"Error: {e}")
     ```

129. **How do you validate DataFrame inputs?**  
     Ensures correct structure.  
     ```python
     def validate_df(df, expected_cols):
         if not all(col in df.columns for col in expected_cols):
             raise ValueError("Missing columns")
         return df
     ```

130. **How do you handle missing data errors in Pandas?**  
     Checks for NaNs before processing.  
     ```python
     def check_missing(df, column):
         if df[column].isna().any():
             raise ValueError(f"Missing values in {column}")
         return df
     ```

131. **What is the role of logging in Pandas debugging?**  
     Tracks errors and operations.  
     ```python
     import logging
     logging.basicConfig(filename='pandas.log', level=logging.INFO)
     logging.info("Starting Pandas operation")
     ```

132. **How do you handle type inconsistencies in Pandas?**  
     Converts to consistent types.  
     ```python
     df['A'] = df['A'].astype(float, errors='ignore')
     ```

#### Intermediate
133. **Write a function to retry Pandas operations on failure.**  
     Handles transient errors.  
     ```python
     def retry_operation(func, df, max_attempts=3):
         for attempt in range(max_attempts):
             try:
                 return func(df)
             except Exception as e:
                 if attempt == max_attempts - 1:
                     raise
                 print(f"Attempt {attempt+1} failed: {e}")
     ```

134. **How do you debug Pandas operation outputs?**  
     Inspects intermediate results.  
     ```python
     def debug_operation(df, column):
         result = df[column].mean()
         print(f"Column: {column}, Mean: {result}")
         return result
     ```

135. **Write a function to validate DataFrame outputs.**  
     Ensures correct results.  
     ```python
     def validate_output(df, column, expected_type):
         if not df[column].dtype == expected_type:
             raise ValueError(f"Expected type {expected_type}, got {df[column].dtype}")
         return df
     ```

136. **How do you profile Pandas operation performance?**  
     Measures execution time.  
     ```python
     import time
     def profile_operation(df, column):
         start = time.time()
         result = df[column].mean()
         print(f"Operation took {time.time() - start}s")
         return result
     ```

137. **Write a function to handle memory errors in Pandas.**  
     Manages large DataFrames.  
     ```python
     def safe_operation(df, max_rows=1e6):
         if len(df) > max_rows:
             raise MemoryError("DataFrame too large")
         return df
     ```

138. **How do you debug Pandas merge errors?**  
     Logs merge issues.  
     ```python
     def debug_merge(df1, df2, key):
         try:
             return pd.merge(df1, df2, on=key)
         except KeyError as e:
             print(f"Merge error: {e}")
             raise
     ```

#### Advanced
139. **Write a function to implement a custom Pandas error handler.**  
     Logs specific errors.  
     ```python
     import logging
     def custom_error_handler(df, operation):
         logging.basicConfig(filename='pandas.log', level=logging.ERROR)
         try:
             return operation(df)
         except Exception as e:
             logging.error(f"Operation error: {e}")
             raise
     ```

140. **How do you implement circuit breakers in Pandas applications?**  
     Prevents cascading failures.  
     ```python
     from pybreaker import CircuitBreaker
     breaker = CircuitBreaker(fail_max=3, reset_timeout=60)
     @breaker
     def safe_operation(df):
         return df.groupby('A').sum()
     ```

141. **Write a function to detect data inconsistencies.**  
     Validates data integrity.  
     ```python
     def detect_inconsistencies(df, column):
         if df[column].isna().any():
             print(f"Warning: Missing values in {column}")
         return df
     ```

142. **How do you implement logging for distributed Pandas operations?**  
     Centralizes logs for debugging.  
     ```python
     import logging.handlers
     def setup_distributed_logging():
         handler = logging.handlers.SocketHandler('log-server', 9090)
         logging.getLogger().addHandler(handler)
         logging.info("Pandas operation started")
     ```

143. **Write a function to handle version compatibility in Pandas.**  
     Checks library versions.  
     ```python
     import pandas as pd
     def check_pandas_version():
         if pd.__version__ < '1.0':
             raise ValueError("Unsupported Pandas version")
     ```

144. **How do you debug Pandas performance bottlenecks?**  
     Profiles operation stages.  
     ```python
     import time
     def debug_bottlenecks(df):
         start = time.time()
         result = df.groupby('A').sum()
         print(f"Grouping: {time.time() - start}s")
         return result
     ```

## Best Practices and Optimization

### Basic
145. **What are best practices for Pandas code organization?**  
     Modularizes data operations.  
     ```python
     def preprocess_data(df):
         return clean_pipeline(df)
     def compute_features(df):
         return df[['A', 'B']].mean()
     ```

146. **How do you ensure reproducibility in Pandas?**  
     Sets random seeds.  
     ```python
     np.random.seed(42)
     ```

147. **What is caching in Pandas pipelines?**  
     Stores processed DataFrames.  
     ```python
     from functools import lru_cache
     @lru_cache(maxsize=1000)
     def preprocess_df(df):
         return clean_pipeline(df)
     ```

148. **How do you handle large-scale Pandas DataFrames?**  
     Uses chunked processing.  
     ```python
     def process_large_df(df, chunk_size=1000):
         chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
         return pd.concat([clean_pipeline(chunk) for chunk in chunks])
     ```

149. **What is the role of environment configuration in Pandas?**  
     Manages settings securely.  
     ```python
     import os
     os.environ['PANDAS_DATA_PATH'] = 'data.csv'
     ```

150. **How do you document Pandas code?**  
     Uses docstrings for clarity.  
     ```python
     def clean_pipeline(df):
         """Cleans DataFrame by removing NaNs and duplicates."""
         return df.dropna().drop_duplicates()
     ```

#### Intermediate
151. **Write a function to optimize Pandas memory usage.**  
     Limits memory allocation.  
     ```python
     def optimize_memory(df):
         df = df.astype({col: 'float32' for col in df.select_dtypes('float64').columns})
         return df
     ```

152. **How do you implement unit tests for Pandas code?**  
     Validates DataFrame operations.  
     ```python
     import unittest
     class TestPandas(unittest.TestCase):
         def test_clean(self):
             df = pd.DataFrame({'A': [1, np.nan]})
             result = clean_pipeline(df)
             self.assertFalse(result.isna().any().any())
     ```

153. **Write a function to create reusable Pandas templates.**  
     Standardizes data processing.  
     ```python
     def process_template(df, operation='clean'):
         if operation == 'clean':
             return clean_pipeline(df)
         return df
     ```

154. **How do you optimize Pandas for batch processing?**  
     Processes DataFrames in chunks.  
     ```python
     def batch_process(dfs, batch_size=100):
         for i in range(0, len(dfs), batch_size):
             yield [clean_pipeline(df) for df in dfs[i:i+batch_size]]
     ```

155. **Write a function to handle Pandas configuration.**  
     Centralizes settings.  
     ```python
     def configure_pandas():
         return {'dtype': 'float32', 'index_col': 'id'}
     ```

156. **How do you ensure Pandas pipeline consistency?**  
     Standardizes versions and settings.  
     ```python
     import pandas as pd
     def check_pandas_env():
         print(f"Pandas version: {pd.__version__}")
     ```

#### Advanced
157. **Write a function to implement Pandas pipeline caching.**  
     Reuses processed DataFrames.  
     ```python
     def cache_df(df, cache_file='cache.pkl'):
         if os.path.exists(cache_file):
             return pd.read_pickle(cache_file)
         result = clean_pipeline(df)
         result.to_pickle(cache_file)
         return result
     ```

158. **How do you optimize Pandas for high-throughput processing?**  
     Uses parallel execution.  
     ```python
     from joblib import Parallel, delayed
     def high_throughput_process(dfs):
         return Parallel(n_jobs=-1)(delayed(clean_pipeline)(df) for df in dfs)
     ```

159. **Write a function to implement Pandas pipeline versioning.**  
     Tracks changes in workflows.  
     ```python
     def version_pipeline(config, version):
         with open(f'pandas_pipeline_v{version}.json', 'w') as f:
             json.dump(config, f)
     ```

160. **How do you implement Pandas pipeline monitoring?**  
     Logs performance metrics.  
     ```python
     import logging
     def monitored_process(df):
         logging.basicConfig(filename='pandas.log', level=logging.INFO)
         start = time.time()
         result = clean_pipeline(df)
         logging.info(f"Processed DataFrame in {time.time() - start}s")
         return result
     ```

161. **Write a function to handle Pandas scalability.**  
     Processes large datasets efficiently.  
     ```python
     def scalable_process(df, chunk_size=1000):
         chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
         return pd.concat([clean_pipeline(chunk) for chunk in chunks])
     ```

162. **How do you implement Pandas pipeline automation?**  
     Scripts end-to-end workflows.  
     ```python
     def automate_pipeline(df):
         processed = clean_pipeline(df)
         processed.to_csv('processed_data.csv')
         return processed
     ```

## Ethical Considerations in Pandas

### Basic
163. **What are ethical concerns in Pandas applications?**  
     Includes bias in data processing and privacy.  
     ```python
     def check_data_bias(df, target_col):
         return df.groupby(target_col).mean().diff().abs()
     ```

164. **How do you detect bias in Pandas data processing?**  
     Analyzes group disparities.  
     ```python
     def detect_bias(df, group_col, value_col):
         return df.groupby(group_col)[value_col].mean()
     ```

165. **What is data privacy in Pandas, and how is it ensured?**  
     Protects sensitive data.  
     ```python
     def anonymize_data(df, column):
         df[column] = df[column].apply(lambda x: hash(x))
         return df
     ```

166. **How do you ensure fairness in Pandas data processing?**  
     Balances data across groups.  
     ```python
     def fair_processing(df, target_col):
         return balance_df(df, target_col)
     ```

167. **What is explainability in Pandas applications?**  
     Clarifies data transformations.  
     ```python
     def explain_transformation(df, column):
         print(f"Before: {df[column].describe()}")
         df[column] = df[column].fillna(df[column].mean())
         print(f"After: {df[column].describe()}")
         return df
     ```

168. **How do you visualize Pandas data bias?**  
     Plots group-wise statistics.  
     ```python
     import matplotlib.pyplot as plt
     def plot_bias(df, group_col, value_col):
         df.groupby(group_col)[value_col].mean().plot(kind='bar')
         plt.savefig('bias_plot.png')
     ```

#### Intermediate
169. **Write a function to mitigate bias in Pandas data.**  
     Reweights or resamples data.  
     ```python
     def mitigate_bias(df, target_col):
         return balance_df(df, target_col)
     ```

170. **How do you implement differential privacy in Pandas?**  
     Adds noise to protect data.  
     ```python
     def private_processing(df, column, epsilon=1.0):
         noise = np.random.laplace(0, 1/epsilon, len(df))
         df[column] += noise
         return df
     ```