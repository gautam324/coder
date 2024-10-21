import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
train = pd.read_csv('/kaggle/input/dataset/train.csv')
test = pd.read_csv('/kaggle/input/dataset/test.csv')

# Feature engineering function
def feature_engineering(df):
    df['price_change'] = df['close'] - df['open']
    df['price_range'] = df['high'] - df['low']
    df['volume_change'] = df['volume'].pct_change()
    df['quote_volume_change'] = df['quote_asset_volume'].pct_change()
    df['trade_ratio'] = df['taker_buy_base_volume'] / df['taker_buy_quote_volume']
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Adding more technical indicators
    df['moving_avg_5'] = df['close'].rolling(window=5).mean()
    df['moving_avg_20'] = df['close'].rolling(window=20).mean()
    df['rsi'] = 100 - (100 / (1 + (df['close'].diff().where(df['close'].diff() > 0, 0).rolling(window=14).mean() / df['close'].diff().where(df['close'].diff() < 0, 0).rolling(window=14).mean())))

    # Additional features
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['bollinger_high'] = df['moving_avg_20'] + 2 * df['close'].rolling(window=20).std()
    df['bollinger_low'] = df['moving_avg_20'] - 2 * df['close'].rolling(window=20).std()
    df['volatility'] = df['log_return'].rolling(window=10).std()

    # Fill missing values
    df.ffill(inplace=True)
    df.fillna(df.mean(), inplace=True)

    return df

# Apply feature engineering
train = feature_engineering(train)
test = feature_engineering(test)

# Features and target
features = train.columns.drop(['timestamp', 'target'])
X = train[features]
y = train['target']

# Stratified split for balanced class distribution
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handling NaN and Inf values
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_val.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(X_train.mean(), inplace=True)
X_val.fillna(X_val.mean(), inplace=True)





# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_val_scaled = scaler.transform(X_val)

# Reshaping the data for LSTM input (3D tensor: [samples, timesteps, features])
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_val_lstm = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))

# Building the LSTM Model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_lstm, y_train_smote, epochs=15, batch_size=64,
                    validation_data=(X_val_lstm, y_val), callbacks=[early_stopping], verbose=2)

# Predictions on validation set
y_pred_val = (model.predict(X_val_lstm) > 0.5).astype(int)

# Calculate F1 score
val_f1_score = f1_score(y_val, y_pred_val)
print(f"Validation F1 Score: {val_f1_score}")
# Handling NaN and Inf values in the test set
test.replace([np.inf, -np.inf], np.nan, inplace=True)
test.fillna(test.mean(), inplace=True)

# Prepare the test set for predictions
test_scaled = scaler.transform(test[features])

# Reshaping the test set for LSTM input
test_lstm = test_scaled.reshape((test_scaled.shape[0], 1, test_scaled.shape[1]))

# Predictions on the test set
test_pred = (model.predict(test_lstm) > 0.5).astype(int)

# Prepare the submission file
submission = pd.DataFrame({
    'row_id': test['row_id'],
    'target': test_pred.flatten().astype(int)
})

# Save the submission
submission.to_csv('submission_lstm.csv', index=False)
