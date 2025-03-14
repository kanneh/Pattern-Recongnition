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
from tensorflow.keras.optimizers import Adam

# Load Dataset2
df2 = pd.read_csv("MM19_CSDB_DS.csdb.csv")

# Handle missing values
df2.ffill(inplace=True)

# Create lagged features for time-series prediction
for col in df2.columns[1:]:
    for lag in range(1, 4):
        df2[f'{col}_lag{lag}'] = df2[col].shift(lag)

df2.dropna(inplace=True)

# Ensure all columns are numeric
df2 = df2.apply(pd.to_numeric, errors='coerce')

# Split data
X = df2.drop('K33V', axis=1)
y = df2['K33V']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Train and evaluate ARIMA
arima = ARIMA(y_train, order=(1, 1, 1))
arima_fit = arima.fit()
y_pred_arima = arima_fit.forecast(steps=len(y_test))

# Train and evaluate LSTM
X_train_lstm = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Check for NaN values
print("NaN in X_train_lstm:", np.isnan(X_train_lstm).sum())
print("NaN in X_test_lstm:", np.isnan(X_test_lstm).sum())
print("NaN in y_train:", np.isnan(y_train).sum())
print("NaN in y_test:", np.isnan(y_test).sum())

# Handle NaN values
X_train_lstm = np.nan_to_num(X_train_lstm)
X_test_lstm = np.nan_to_num(X_test_lstm)
y_train = np.nan_to_num(y_train)
y_test = np.nan_to_num(y_test)

# Build and train LSTM model
lstm = Sequential()
lstm.add(LSTM(10, activation='relu', input_shape=(X_train_lstm.shape[1], 1)))
lstm.add(Dense(1))
lstm.compile(optimizer=Adam(clipvalue=1.0), loss='mse')
lstm.fit(X_train_lstm, y_train, epochs=50, verbose=0)
y_pred_lstm = lstm.predict(X_test_lstm).flatten()

# Check for NaN in predictions
print("NaN in y_pred_lstm:", np.isnan(y_pred_lstm).sum())

# Evaluate models
print("ARIMA - MAE:", mean_absolute_error(y_test, y_pred_arima), "RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_arima)))
print("LSTM - MAE:", mean_absolute_error(y_test, y_pred_lstm), "RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lstm)))
print("Random Forest - MAE:", mean_absolute_error(y_test, y_pred_rf), "RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))