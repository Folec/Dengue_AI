import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os
import pickle

class DengueLSTM:
    def __init__(self, data_dir='Data', city='sj', lags=35, hidden_size=64, num_layers=2, epochs=50, patience=5, seed=42):
        # Ensure data_dir is set relative to the project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.data_dir = os.path.join(project_root, data_dir)
        
        self.city = city
        self.lags = lags
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self._set_seed()
        self._load_data()
        self._prepare_data()
        self._build_model()

    def _set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        import random
        random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_data(self):
        # Use os.path.join for cross-platform compatibility
        features_train_path = os.path.join(self.data_dir, 'dengue_features_train.csv')
        labels_train_path = os.path.join(self.data_dir, 'dengue_labels_train.csv')
        features_test_path = os.path.join(self.data_dir, 'dengue_features_test.csv')

        self.x_train = pd.read_csv(features_train_path, index_col=[0, 1, 2])
        self.y_train = pd.read_csv(labels_train_path, index_col=[0, 1, 2])
        self.x_test = pd.read_csv(features_test_path, index_col=[0, 1, 2])

    def _prepare_data(self):
        # Select city data
        sj_x_train = self.x_train.loc[self.city].copy()
        sj_y_train = self.y_train.loc[self.city].copy()
        sj_x_test = self.x_test.loc[self.city].copy()
        sj_x_train.ffill(inplace=True)
        sj_x_test.ffill(inplace=True)
        sj_x_train.drop('week_start_date', axis=1, inplace=True)

        # Feature selection
        sj_x_train['total_cases'] = sj_y_train['total_cases']
        sj_corr = sj_x_train.corr()
        self.sj_top_features = sj_corr['total_cases'].drop('total_cases').abs().sort_values(ascending=False).head(9).index.tolist()
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
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(x_train_sj[self.sj_top_features])
        X_test_scaled = self.feature_scaler.transform(x_test_sj[self.sj_top_features])
        y_train_scaled = self.target_scaler.fit_transform(y_train_sj['total_cases'].values.reshape(-1, 1)).flatten()
        y_test_scaled = self.target_scaler.transform(y_test_sj['total_cases'].values.reshape(-1, 1)).flatten()

        # Lagged sequences
        def lagged_sequences_np(X, y, lags):
            Xs, ys = [], []
            for i in range(lags, len(X)):
                Xs.append(X[i-lags:i])
                ys.append(y[i])
            return np.array(Xs), np.array(ys)

        self.X_train_seq, self.y_train_seq = lagged_sequences_np(X_train_scaled, y_train_scaled, self.lags)
        self.X_test_seq, self.y_test_seq = lagged_sequences_np(X_test_scaled, y_test_scaled, self.lags)

        self.X_train_torch = torch.tensor(self.X_train_seq, dtype=torch.float32)
        self.y_train_torch = torch.tensor(self.y_train_seq, dtype=torch.float32)
        self.X_test_torch = torch.tensor(self.X_test_seq, dtype=torch.float32)
        self.y_test_torch = torch.tensor(self.y_test_seq, dtype=torch.float32)

        # For prediction on unseen test set
        self.sj_x_test = sj_x_test

    class LSTMRegressor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    def _build_model(self):
        self.model = self.LSTMRegressor(
            input_size=self.X_train_torch.shape[2],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=0.01)

    def train(self):
        best_mae = float('inf')
        best_epoch = 0
        best_model_state = None
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(self.X_train_torch).squeeze()
            loss = self.criterion(output, self.y_train_torch.squeeze())
            loss.backward()
            self.optimizer.step()

            # Evaluate on test set at every epoch
            self.model.eval()
            with torch.no_grad():
                y_pred_scaled = self.model(self.X_test_torch).squeeze().numpy()
                y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                y_true = self.target_scaler.inverse_transform(self.y_test_torch.squeeze().numpy().reshape(-1, 1)).flatten()
                y_pred = np.clip(y_pred, 0, None)
                mae = np.mean(np.abs(y_pred - y_true))

                # Early stopping check
                if mae < best_mae:
                    best_mae = mae
                    best_epoch = epoch
                    best_model_state = self.model.state_dict()
                elif epoch - best_epoch >= self.patience:
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def save_model(self, model_path=None, scaler_path=None):
        """Save the trained model and scalers"""
        if model_path is None:
            model_path = f'dengue_lstm_{self.city}.pth'
        if scaler_path is None:
            scaler_path = f'dengue_scalers_{self.city}.pkl'
        
        # Save model state dict
        torch.save(self.model.state_dict(), model_path)
        
        # Save scalers and other necessary components
        model_components = {
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'sj_top_features': self.sj_top_features,
            'lags': self.lags,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'input_size': self.X_train_torch.shape[2]
        }
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(model_components, f)
        
        print(f"Model saved to {model_path}")
        print(f"Model components saved to {scaler_path}")