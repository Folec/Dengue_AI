import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import os

class DengueLSTM:
    # City-specific optimal hyperparameters (from your notebook)
    CITY_PARAMS = {
        'sj': dict(lags=35, hidden_size=64, num_layers=2, epochs=50, patience=5),
        'iq': dict(lags=20, hidden_size=64, num_layers=2, epochs=50, patience=5),  # Use your optimal IQ params here
    }

    def __init__(self, data_dir='Data', city='sj', seed=42):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.data_dir = os.path.join(project_root, data_dir)
        self.city = city
        self.seed = seed

        # Set city-specific hyperparameters
        params = self.CITY_PARAMS[city]
        self.lags = params['lags']
        self.hidden_size = params['hidden_size']
        self.num_layers = params['num_layers']
        self.epochs = params['epochs']
        self.patience = params['patience']

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
        features_train_path = os.path.join(self.data_dir, 'dengue_features_train.csv')
        labels_train_path = os.path.join(self.data_dir, 'dengue_labels_train.csv')
        features_test_path = os.path.join(self.data_dir, 'dengue_features_test.csv')

        self.x_train = pd.read_csv(features_train_path, index_col=[0, 1, 2])
        self.y_train = pd.read_csv(labels_train_path, index_col=[0, 1, 2])
        self.x_test = pd.read_csv(features_test_path, index_col=[0, 1, 2])

    def _prepare_data(self):
        x_train_city = self.x_train.loc[self.city].copy()
        y_train_city = self.y_train.loc[self.city].copy()
        x_test_city = self.x_test.loc[self.city].copy()
        x_train_city.ffill(inplace=True)
        x_test_city.ffill(inplace=True)
        x_train_city.drop('week_start_date', axis=1, inplace=True)

        # Feature selection: use correlation with total_cases
        x_train_city['total_cases'] = y_train_city['total_cases']
        corr = x_train_city.corr()
        if self.city == 'sj':
            top_features = corr['total_cases'].drop('total_cases').abs().sort_values(ascending=False).head(9).index.tolist()
        else:
            top_features = corr['total_cases'].drop('total_cases').abs().sort_values(ascending=False).head(4).index.tolist()
        x_train_city.drop('total_cases', axis=1, inplace=True)
        self.top_features = top_features

        # TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        for train_indices, test_indices in tscv.split(x_train_city):
            pass
        x_train_final = x_train_city.iloc[train_indices].reset_index(drop=True)
        x_test_final = x_train_city.iloc[test_indices].reset_index(drop=True)
        y_train_final = y_train_city.iloc[train_indices].reset_index(drop=True)
        y_test_final = y_train_city.iloc[test_indices].reset_index(drop=True)

        # Scaling
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        X_train_scaled = self.feature_scaler.fit_transform(x_train_final[self.top_features])
        X_test_scaled = self.feature_scaler.transform(x_test_final[self.top_features])
        y_train_scaled = self.target_scaler.fit_transform(y_train_final['total_cases'].values.reshape(-1, 1)).flatten()
        y_test_scaled = self.target_scaler.transform(y_test_final['total_cases'].values.reshape(-1, 1)).flatten()

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
        self.x_test_city = x_test_city

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

    def predict_unseen(self):
        x_pred_city = self.x_test_city[self.top_features].copy()
        x_pred_city_scaled = self.feature_scaler.transform(x_pred_city)

        def lagged_sequences_predict(X, lags):
            Xs = []
            for i in range(lags, len(X)):
                Xs.append(X[i-lags:i])
            return np.array(Xs)

        x_pred_city_seq = lagged_sequences_predict(x_pred_city_scaled, self.lags)
        X_pred_city_torch = torch.tensor(x_pred_city_seq, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(X_pred_city_torch).squeeze().numpy()
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_pred = np.round(y_pred).astype(int)
            y_pred = np.clip(y_pred, 0, None)
        return y_pred

    def create_city_submission(self):
        y_pred = self.predict_unseen()
        dates = self.x_test_city.index[self.lags:]
        df = pd.DataFrame({
            'city': self.city,
            'year': [idx[0] for idx in dates],
            'weekofyear': [idx[1] for idx in dates],
            'total_cases': y_pred
        })
        return df

    def get_city_dataframes(self):
        dfs = []
        for city in ['sj', 'iq']:
            model = DengueLSTM(city=city)
            model.train()
            df = model.create_city_submission()
            dfs.append(df)
            del model
            import gc
            gc.collect()
        submission = pd.concat(dfs, ignore_index=True)
        print(submission.to_csv(index=False))
        return submission

# To run for both cities and print the CSV:
DengueLSTM().get_city_dataframes()