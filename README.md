# Big Mart Product Outlet Sales Mini Project

### Overview
This mini project aims to analyze the Big Mart Product Outlet Sales dataset to understand the properties of products and stores that influence sales. The project involves exploratory data analysis (EDA), feature engineering, hypothesis generation, and model building to identify factors that contribute to increased sales.

### Specification and Requirements
- Carry out exploratory data analysis (EDA) to understand the dataset's structure and characteristics.
- Identify properties of products and stores that influence sales.
- Test statistical hypotheses to validate assumptions and findings.

### Description of the Data
The dataset contains sales data for the year 2013, encompassing 1559 products across 10 stores located in different cities. Each data entry includes the following columns:

- **Item_Identifier**: Unique identifier for each product.
- **Item_Weight**: Weight of the product.
- **Item_Fat_Content**: Fat content of the product.
- **Item_Visibility**: Percentage of total display area allocated to the product.
- **Item_Type**: Category of the product.
- **Item_MRP**: Maximum retail price of the product.
- **Outlet_Identifier**: Unique identifier for each store.
- **Outlet_Size**: Size of the store.
- **Outlet_Location_Type**: Type of location where the store is situated.
- **Outlet_Type**: Type of the store.

### Project Structure
The project consists of the following files:
1. **hypothesis_generation.ipynb**: Notebook containing the process of generating hypotheses based on the dataset.
2. **EDA & Feature engineering.ipynb**: Notebook showcasing exploratory data analysis and feature engineering techniques applied to the dataset.
3. **modeling.py**: Python script implementing machine learning models to predict sales based on the dataset.

### Dependencies
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

#### Data Analysis and Wrangling:
- Pandas
- NumPy

#### Visualization:
- Seaborn
- Matplotlib

#### Preprocessing:
- Pandas
- Scikit-learn
  - StandardScaler
  - OneHotEncoder

#### Modeling:
- Scikit-learn
  - LinearRegression
  - Lasso
  - RandomForestRegressor
  - train_test_split
  - mean_squared_error
  - r2_score
- XGBoost

#### Hypothesis Testing:
- SciPy
  - pearsonr
- Statsmodels
  - ols
- Scipy.stats
  - ttest_ind

Make sure to install these dependencies before running the project.
