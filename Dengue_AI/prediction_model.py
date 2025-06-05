import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error


# Data Loading 
x_train = pd.read_csv('Data/dengue_features_train.csv', index_col=[0, 1, 2])
y_train = pd.read_csv('Data/dengue_labels_train.csv', index_col=[0, 1, 2])
x_test = pd.read_csv('Data/dengue_features_test.csv', index_col=[0, 1, 2])

# Select San Juan data 
sj_x_train = x_train.loc['sj'].copy()
sj_y_train = y_train.loc['sj'].copy()
sj_x_test = x_test.loc['sj'].copy()

sj_x_train.ffill(inplace=True)
sj_x_test.ffill(inplace=True)
sj_x_train.drop('week_start_date', axis=1, inplace=True)

# Feature selection
sj_x_train['total_cases'] = sj_y_train['total_cases']
sj_corr = sj_x_train.corr()
sj_top_features = sj_corr['total_cases'].drop('total_cases').abs().sort_values(ascending=False).head(9).index.tolist()
sj_x_train.drop('total_cases', axis=1, inplace=True)

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_indices, test_indices in tscv.split(sj_x_train):
    pass

x_train_sj = sj_x_train.iloc[train_indices].reset_index(drop=True)
x_test_sj = sj_x_train.iloc[test_indices].reset_index(drop=True)
y_train_sj = sj_y_train.iloc[train_indices].reset_index(drop=True)
y_test_sj = sj_y_train.iloc[test_indices].reset_index(drop=True)

# Scaling 
feature_scaler = StandardScaler()
target_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(x_train_sj[sj_top_features])
X_test_scaled = feature_scaler.transform(x_test_sj[sj_top_features])
y_train_scaled = target_scaler.fit_transform(y_train_sj['total_cases'].values.reshape(-1, 1)).flatten()
y_test_scaled = target_scaler.transform(y_test_sj['total_cases'].values.reshape(-1, 1)).flatten()

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import random
random.seed(42)

# Lagged sequences (same as notebook)
def lagged_sequences_np(X, y, lags):
    Xs, ys = [], []
    for i in range(lags, len(X)):
        Xs.append(X[i-lags:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

lags = 7
hidden_size = 16
num_layers = 1

X_train_seq, y_train_seq = lagged_sequences_np(X_train_scaled, y_train_scaled, lags)
X_test_seq, y_test_seq = lagged_sequences_np(X_test_scaled, y_test_scaled, lags)

X_train_torch = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_torch = torch.tensor(y_train_seq, dtype=torch.float32)
X_test_torch = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_torch = torch.tensor(y_test_seq, dtype=torch.float32)

# LSTM Model (same as notebook) 
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LSTMRegressor(
    input_size=X_train_torch.shape[2],
    hidden_size=hidden_size,
    num_layers=num_layers
)

criterion = nn.L1Loss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
epochs = 50
patience = 5

mae_history = []
best_mae = float('inf')
best_epoch = 0
best_model_state = None

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_torch).squeeze()
    loss = criterion(output, y_train_torch.squeeze())
    loss.backward()
    optimizer.step()
    
    # Evaluate on test set at every epoch
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_torch).squeeze().numpy()
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = target_scaler.inverse_transform(y_test_torch.squeeze().numpy().reshape(-1, 1)).flatten()
        y_pred = np.clip(y_pred, 0, None)
        mae = np.mean(np.abs(y_pred - y_true))
        mae_history.append(mae)
        
        # Early stopping check
        if mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            best_model_state = model.state_dict()
        elif epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best MAE: {best_mae:.4f}")
            break

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test MAE: {mae:.4f}")

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Restored best model from epoch {best_epoch+1} with MAE: {best_mae:.4f}")

print("Final LSTM MAE on test set:", best_mae)

# Predict on unseen test set
x_pred_sj = sj_x_test[sj_top_features].copy()
x_pred_sj_scaled = feature_scaler.transform(x_pred_sj)

def lagged_sequences_predict(X, lags):
    Xs = []
    for i in range(lags, len(X)):
        Xs.append(X[i-lags:i])
    return np.array(Xs)

x_pred_sj_seq = lagged_sequences_predict(x_pred_sj_scaled, lags)
X_pred_sj_torch = torch.tensor(x_pred_sj_seq, dtype=torch.float32)

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_pred_sj_torch).squeeze().numpy()
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_pred = np.round(y_pred).astype(int)
    y_pred = np.clip(y_pred, 0, None)
print("Predictions on unseen test set:", y_pred)