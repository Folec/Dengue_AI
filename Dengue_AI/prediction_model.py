import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

class linear_regression:    
    def __init__(self) :
        self.model = None
        self.X_final = None
        self.feature_names = None

    def create_X_final():
        np.random.seed(42)
        n_samples = 500

        model = LinearRegression()

        villes = ['Bruxelles', 'Gand', 'Anvers', 'Liège', 'Namur', 'Louvain']
        heures = np.random.randint(0, 24, n_samples)
        jours = np.random.choice(['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'], n_samples)
        villes_sample = np.random.choice(villes, n_samples)
        temperature = np.random.uniform(15, 30, n_samples)
        humidite = np.random.uniform(30, 90, n_samples)
        precipitations = np.random.uniform(0, 10, n_samples)
        vent = np.random.uniform(0, 20, n_samples)
        pression = np.random.uniform(980, 1050, n_samples)
        ensoleillement = np.random.uniform(0, 12, n_samples)
        visibilite = np.random.uniform(1, 10, n_samples)
        pollution = np.random.uniform(0, 100, n_samples)
        nebulosite = np.random.uniform(0, 100, n_samples)
        neige = np.random.uniform(0, 50, n_samples)
        orage = np.random.choice([0, 1], n_samples)

        valeur = (
            heures * 2.5 +
            np.array([villes.index(v) for v in villes_sample]) * 5 +
            np.random.normal(0, 10, n_samples)
        )

        df = pd.DataFrame({
            'ville': villes_sample,
            'heure': heures,
            'jour': jours,
            'temperature': temperature,
            'humidite': humidite,
            'precipitations': precipitations,
            'vent': vent,
            'pression': pression,
            'ensoleillement': ensoleillement,
            'visibilite': visibilite,
            'pollution': pollution,
            'nebulosite': nebulosite,
            'neige': neige,
            'orage': orage,
            'valeur': valeur
        })

        X = df[['ville', 'heure', 'jour', 'temperature', 'humidite', 'precipitations', 'vent', 'pression',
                'ensoleillement', 'visibilite', 'pollution', 'nebulosite', 'neige', 'orage']]
        y = df['valeur']

        encoder = OneHotEncoder(sparse_output=False)
        X_encoded = encoder.fit_transform(X[['ville', 'jour']])
        X_final = np.concatenate([X[['heure']].values, X_encoded], axis=1)

        # Generate feature names for encoded data
        categorical_feature_names = encoder.get_feature_names_out(['ville', 'jour'])
        feature_names = ['heure'] + list(categorical_feature_names)

        model = LinearRegression()
        model.fit(X_final, y)

        # 4. Prédiction
        df['prediction'] = model.predict(X_final)

        return X_final, feature_names, model, df

## Load the data  
x_train = pd.read_csv('Data/DengAI_Predicting_Disease_Spread_-_Training_Data_Features.csv', index_col= [0, 1, 2])
y_train = pd.read_csv('Data/DengAI_Predicting_Disease_Spread_-_Training_Data_Labels.csv', index_col= [0, 1, 2])
x_test = pd.read_csv('Data/DengAI_Predicting_Disease_Spread_-_Test_Data_Features.csv', index_col= [0, 1, 2])

sj_x_train = x_train.loc['sj'].copy()
iq_x_train = x_train.loc['iq'].copy()

sj_y_train = y_train.loc['sj'].copy()
iq_y_train = y_train.loc['iq'].copy()  

sj_x_test = x_test.loc['sj'].copy()
iq_x_test = x_test.loc['iq'].copy()

sj_x_train.ffill(inplace=True)
sj_x_test.ffill(inplace=True)

iq_x_train.ffill(inplace=True)
iq_x_test.ffill(inplace=True)


# Time Series Split for San Juan and Iquitos
from sklearn.model_selection import TimeSeriesSplit

# For San Juan
tscv = TimeSeriesSplit(n_splits=5)
for train_indices, test_indices in tscv.split(sj_x_train):
    pass  # This will leave train_indices and test_indices as the last split

x_train_sj = sj_x_train.iloc[train_indices].reset_index(drop=True)
x_test_sj = sj_x_train.iloc[test_indices].reset_index(drop=True)
y_train_sj = sj_y_train.iloc[train_indices].reset_index(drop=True)
y_test_sj = sj_y_train.iloc[test_indices].reset_index(drop=True)

# For Iquitos
tscv = TimeSeriesSplit(n_splits=5)
for train_indices, test_indices in tscv.split(iq_x_train):
    pass

x_train_iq = iq_x_train.iloc[train_indices].reset_index(drop=True)
x_test_iq = iq_x_train.iloc[test_indices].reset_index(drop=True)
y_train_iq = iq_y_train.iloc[train_indices].reset_index(drop=True)
y_test_iq = iq_y_train.iloc[test_indices].reset_index(drop=True)

## LSTM model for San Juan 
import torch
import torch.nn as nn
import numpy as np

# Calculate sj_top_features again using only the training set (to avoid data leakage)
correlations = x_train_sj.corrwith(y_train_sj)
sj_top_features = correlations.abs().sort_values(ascending=False).head(10).index.tolist()

class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step
        out = self.fc(out)
        return out

def lagged_sequences(X, y, lags):
    Xs, ys = [], []
    for i in range(lags, len(X)):
        Xs.append(X.iloc[i-lags:i].values)
        ys.append(y.iloc[i].values)
    return np.array(Xs), np.array(ys)

lags = 35
hidden_size = 64
num_layers = 2

X_train_seq, y_train_seq = lagged_sequences(x_train_sj[sj_top_features], y_train_sj, lags)
X_test_seq, y_test_seq = lagged_sequences(x_test_sj[sj_top_features], y_test_sj, lags)

X_train_torch = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_torch = torch.tensor(y_train_seq, dtype=torch.float32)
X_test_torch = torch.tensor(X_test_seq, dtype=torch.float32)
y_test_torch = torch.tensor(y_test_seq, dtype=torch.float32)

model = LSTMRegressor(
    input_size=X_train_torch.shape[2],
    hidden_size=hidden_size,
    num_layers=num_layers
)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 150


mae_history = []

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
        y_pred = model(X_test_torch).squeeze().numpy()
        y_pred = np.round(y_pred).astype(int)
        y_pred = np.clip(y_pred, 0, None)
        mae = np.mean(np.abs(y_pred - y_test_torch.squeeze().numpy()))
        mae_history.append(mae)
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test MAE: {mae:.4f}")

print("Final LSTM MAE on test set:", mae_history[-1])


## LSTM model for Iquitos
# Calculate iq_top_features again using only the training set (to avoid data leakage)
correlations = x_train_iq.corrwith(y_train_iq)
iq_top_features = correlations.abs().sort_values(ascending=False).head(10).index.tolist()


lags = 20
hidden_size = 64
num_layers = 2

# Prepare lagged sequences for IQ
X_train_seq_iq, y_train_seq_iq = lagged_sequences(x_train_iq[iq_top_features], y_train_iq, lags)
X_test_seq_iq, y_test_seq_iq = lagged_sequences(x_test_iq[iq_top_features], y_test_iq, lags)

X_train_torch_iq = torch.tensor(X_train_seq_iq, dtype=torch.float32)
y_train_torch_iq = torch.tensor(y_train_seq_iq, dtype=torch.float32)
X_test_torch_iq = torch.tensor(X_test_seq_iq, dtype=torch.float32)
y_test_torch_iq = torch.tensor(y_test_seq_iq, dtype=torch.float32)

model_iq = LSTMRegressor(
    input_size=X_train_torch_iq.shape[2],
    hidden_size=hidden_size,
    num_layers=num_layers
)
criterion_iq = nn.L1Loss()
optimizer_iq = torch.optim.Adam(model_iq.parameters(), lr=0.001)

epochs = 40
mae_history_iq = []
raw_mae_history_iq = []

for epoch in range(epochs):
    model_iq.train()
    optimizer_iq.zero_grad()
    output_iq = model_iq(X_train_torch_iq).squeeze()
    loss_iq = criterion_iq(output_iq, y_train_torch_iq.squeeze())
    loss_iq.backward()
    optimizer_iq.step()
    
    # Evaluate on test set at every epoch
    model_iq.eval()
    with torch.no_grad():
        y_pred_iq = model_iq(X_test_torch_iq).squeeze().numpy()
        raw_mae_iq = np.mean(np.abs(y_pred_iq - y_test_torch_iq.squeeze().numpy()))  # Raw MAE
        raw_mae_history_iq.append(raw_mae_iq)

        y_pred_rounded_iq = np.round(y_pred_iq).astype(int)
        y_pred_clipped_iq = np.clip(y_pred_rounded_iq, 0, None)
        mae_iq = np.mean(np.abs(y_pred_clipped_iq - y_test_torch_iq.squeeze().numpy()))  # Final MAE
        mae_history_iq.append(mae_iq)
    
    if (epoch + 1) % 5 == 0:
        print(f"[IQ] Epoch {epoch+1}/{epochs}, Loss: {loss_iq.item():.4f}, Raw MAE: {raw_mae_iq:.4f}, Final MAE: {mae_iq:.4f}")

print("Final Iquitos LSTM Raw MAE on test set:", raw_mae_history_iq[-1])
print("Final Iquitos LSTM MAE on test set:", mae_history_iq[-1])