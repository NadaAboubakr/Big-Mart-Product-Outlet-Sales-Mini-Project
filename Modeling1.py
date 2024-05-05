import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


def linear_regression(train, test):
    xtrain = train.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier','Item_Type','Outlet_Type','Item_MRP_qcut'], axis=1)
    ytrain = train['Item_Outlet_Sales'] 
    xtest = test.drop(['Item_Identifier','Outlet_Identifier','Item_Type','Outlet_Type','Item_MRP_qcut'], axis=1)
    
    model = LinearRegression().fit(xtrain, ytrain)
    y_pred = model.predict(xtest)
    
    r_sq = model.score(xtrain, ytrain)
    
    print("Linear Regression:")
    print(f"Coefficient of determination: {r_sq}")
    print(f"Intercept: {model.intercept_}")
    print(f"Coefficients: {model.coef_}")
    print()


def regularized_linear_regression(train, test):
    X = train.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier','Item_Type','Outlet_Type','Item_MRP_qcut'], axis=1)
    y = train['Item_Outlet_Sales']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    
    y_pred = lasso.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = lasso.score(X_val, y_val)
    
    print("Regularized Linear Regression:")
    print("MSE:", mse)
    print("R2:", r2)
    print()


def random_forest(train, test):
    X = train.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier','Item_Type','Outlet_Type','Item_MRP_qcut'], axis=1)
    y = train['Item_Outlet_Sales']
    X_test = test.drop(['Item_Identifier','Outlet_Identifier','Item_Type','Outlet_Type','Item_MRP_qcut'], axis=1)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_val)
    
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print("Random Forest:")
    print('Mean Squared Error:', mse)
    print('R-squared:', r2)
    print()


def xgboost_model(train, test):
    X = train.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier','Item_Type','Outlet_Type','Item_MRP_qcut'], axis=1)
    y = train['Item_Outlet_Sales']
    X_test = test.drop(['Item_Identifier','Outlet_Identifier','Item_Type','Outlet_Type','Item_MRP_qcut'], axis=1)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, objective="reg:squarederror", random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print("XGBoost:")
    print("MSE:", mse)
    print("R-squared:", r2)
    print()


# Load data
train = pd.read_csv("BMPO_train.csv")
test = pd.read_csv("BMPO_test.csv")

# Call and print metrics for each model
linear_regression(train, test)
regularized_linear_regression(train, test)
random_forest(train, test)
xgboost_model(train, test)
