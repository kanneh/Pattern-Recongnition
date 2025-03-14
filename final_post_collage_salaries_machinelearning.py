import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load Dataset1
df1 = pd.read_csv("final-post-college-salaries.csv")

df1['Early Career Pay'] = df1['Early Career Pay'].replace('[\$,]', '', regex=True).astype(float)
df1['Mid-Career Pay'] = df1['Mid-Career Pay'].replace('[\$,]', '', regex=True).astype(float)
df1['% High Meaning'] = df1['% High Meaning'].replace('[%,]', '', regex=True).replace('[-]', '0', regex=True).astype(float)

# Encode categorical variables
df1 = pd.get_dummies(df1, columns=['Major', 'Degree Type'], drop_first=True)
# Normalize numerical features
scaler = StandardScaler()
df1[['Early Career Pay', '% High Meaning']] = scaler.fit_transform(df1[['Early Career Pay', '% High Meaning']])

# Split data
X = df1.drop('Mid-Career Pay', axis=1)
y = df1['Mid-Career Pay']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# XGBoost Regressor
xgb = XGBRegressor(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Evaluate Linear Regression
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Evaluate Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Evaluate XGBoost
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print("Linear Regression - MAE:", mae_lr, "MSE:", mse_lr, "R²:", r2_lr)
print("Random Forest - MAE:", mae_rf, "MSE:", mse_rf, "R²:", r2_rf)
print("XGBoost - MAE:", mae_xgb, "MSE:", mse_xgb, "R²:", r2_xgb)