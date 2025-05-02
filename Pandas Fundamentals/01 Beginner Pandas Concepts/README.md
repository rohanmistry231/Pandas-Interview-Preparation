# 🌱 Beginner Pandas Concepts (`pandas`)

## 📖 Introduction
Pandas is a powerful Python library for data manipulation and analysis, foundational for AI and machine learning (ML) data preprocessing. Built on NumPy, it enables efficient handling of structured data through DataFrames and Series. This section introduces the fundamentals of Pandas, focusing on **DataFrame and Series Creation**, **Basic Operations**, **Initial Data Cleaning**, and **Data Visualization**, with practical examples and interview insights tailored to beginners in AI/ML.

## 🎯 Learning Objectives
- Create and explore Pandas DataFrames and Series for ML datasets.
- Perform indexing, filtering, sorting, and ranking operations.
- Clean data by handling missing values, duplicates, and type conversions.
- Visualize datasets to uncover patterns and insights for ML.

## 🔑 Key Concepts
- **DataFrame and Series Creation**:
  - Create DataFrames (`pd.DataFrame`, `pd.read_csv`) and Series (`pd.Series`).
  - Import/export data (CSV, JSON).
  - Explore data (`head`, `info`, `describe`).
- **Basic Operations**:
  - Index and select data (`loc`, `iloc`, column selection).
  - Filter rows (`query`, boolean indexing).
  - Sort and rank (`sort_values`, `rank`).
- **Initial Data Cleaning**:
  - Handle missing values (`isna`, `fillna`, `dropna`).
  - Remove duplicates (`drop_duplicates`).
  - Convert data types (`astype`).
- **Data Visualization**:
  - Plot with Pandas (`plot`, `hist`, `box`).
  - Customize visualizations with Matplotlib.

## 📝 Example Walkthroughs
The following Python files demonstrate each subsection:

1. **`dataframe_series_creation.py`**:
   - Creates DataFrames from Iris data (`pd.DataFrame`) and NumPy arrays.
   - Builds Series from lists and DataFrame columns (`pd.Series`).
   - Imports/exports CSV/JSON and explores data (`head`, `info`).
   - Visualizes sepal length distribution (histogram).

   Example code:
   ```python
   import pandas as pd
   df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
   df.to_csv('iris_data.csv')
   print(df.head())
   ```

2. **`basic_operations.py`**:
   - Selects columns and rows (`loc`, `iloc`) from Iris data.
   - Filters rows with boolean indexing and `query`.
   - Sorts DataFrame and ranks features (`sort_values`, `rank`).
   - Visualizes filtered data (scatter plot).

   Example code:
   ```python
   import pandas as pd
   long_sepal = df[df['sepal length (cm)'] > 6.0]
   sorted_df = df.sort_values('sepal length (cm)')
   ```

3. **`initial_data_cleaning.py`**:
   - Handles missing values with `fillna` and `dropna`.
   - Removes duplicates (`drop_duplicates`).
   - Converts data types (`astype`, categorical bins).
   - Visualizes missing values before/after cleaning (bar plot).

   Example code:
   ```python
   import pandas as pd
   df_filled = df.fillna(df.mean())
   df_no_duplicates = df.drop_duplicates()
   ```

4. **`data_visualization.py`**:
   - Creates histograms and box plots with Pandas’ `plot`.
   - Customizes scatter plots and combined histograms with Matplotlib.
   - Visualizes feature correlations and ML dataset classes.
   - Saves plots (e.g., correlation heatmap).

   Example code:
   ```python
   import pandas as pd
   df['sepal length (cm)'].plot.hist(bins=20)
   corr_matrix = df.corr()
   plt.imshow(corr_matrix, cmap='coolwarm')
   ```

## 🛠️ Practical Tasks
1. **DataFrame Creation**:
   - Load a CSV file into a DataFrame and print its summary (`info`, `describe`).
   - Create a Series from a list of ML labels and export to JSON.
2. **Basic Operations**:
   - Filter a DataFrame for samples with a feature above the mean.
   - Sort a DataFrame by a numeric column and rank another column.
3. **Data Cleaning**:
   - Fill missing values in a DataFrame with column medians.
   - Remove duplicate rows and convert a float column to integer.
4. **Visualization**:
   - Plot a histogram of a feature using Pandas’ `plot.hist`.
   - Create a scatter plot of two features with Matplotlib.

## 💡 Interview Tips
- **Common Questions**:
  - How do you create a DataFrame from a CSV file?
  - What’s the difference between `loc` and `iloc`?
  - How do you handle missing values in Pandas for ML?
  - Why visualize data before training an ML model?
- **Tips**:
  - Explain `loc` (label-based) vs. `iloc` (index-based) with examples.
  - Highlight imputation (`fillna`) vs. dropping (`dropna`) for missing values.
  - Be ready to code filtering or visualization tasks (e.g., `df[df['col'] > value]`).
- **Coding Tasks**:
  - Load and summarize a dataset with Pandas.
  - Filter rows based on a condition and sort the results.
  - Clean missing values and visualize a feature distribution.

## 📚 Resources
- [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/index.html)
- [10 Minutes to Pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Pandas User Guide: Basics](https://pandas.pydata.org/docs/user_guide/basics.html)
- [Pandas IO Tools](https://pandas.pydata.org/docs/user_guide/io.html)
- [Pandas Visualization](https://pandas.pydata.org/docs/user_guide/visualization.html)
- [Kaggle: Pandas Course](https://www.kaggle.com/learn/pandas)
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781491957653/)