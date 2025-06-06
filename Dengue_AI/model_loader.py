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

    def load_model(self, model_path, scaler_path):
        """Load a saved model and scalers"""
        with open(scaler_path, 'rb') as f:
            components = pickle.load(f)
        self.feature_scaler = components['feature_scaler']
        self.target_scaler = components['target_scaler']
        self.sj_top_features = components['sj_top_features']
        self.lags = components['lags']
        self.hidden_size = components['hidden_size']
        self.num_layers = components['num_layers']
        input_size = components['input_size']
        self.model = self.LSTMRegressor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def predict_unseen(self):
        x_pred_sj = self.sj_x_test[self.sj_top_features].copy()
        x_pred_sj_scaled = self.feature_scaler.transform(x_pred_sj)

        def lagged_sequences_predict(X, lags):
            Xs = []
            for i in range(lags, len(X)):
                Xs.append(X[i-lags:i])
            return np.array(Xs)

        x_pred_sj_seq = lagged_sequences_predict(x_pred_sj_scaled, self.lags)
        X_pred_sj_torch = torch.tensor(x_pred_sj_seq, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            y_pred_scaled = self.model(X_pred_sj_torch).squeeze().numpy()
            y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_pred = np.round(y_pred).astype(int)
            y_pred = np.clip(y_pred, 0, None)
        return y_pred

    def create_X_final(self):
        y_pred = self.predict_unseen()
        dates = self.sj_x_test.index[self.lags:]
        # Get the features used for prediction, aligned with the prediction dates
        features_df = self.sj_x_test.loc[dates, self.sj_top_features].copy()
        
        # DataFrame with predictions
        df = pd.DataFrame({
            'predicted_cases': y_pred,
        }, index=dates)
        df = df.reset_index()
        features_df = features_df.reset_index()

        df_full = pd.concat([df, features_df.drop(columns=['year', 'weekofyear'], errors='ignore')], axis=1)
        return self.lags, self.sj_top_features, self.model, df_full
    
    def transform_X_final(self, X_final_tuple):
        X_final, feature_names, model, df = X_final_tuple
        return df
    
    def get_city_dataframes(self):
        city_names = {'sj': 'San Juan', 'iq': 'Iquitos'}
        dfs = {}
        for x in ['sj', 'iq']:
            print(f"Loading model for {city_names[x]}...")
            model = DengueLSTM(city=x)
            models_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "Models")
            model_path = os.path.join(models_dir, f'dengue_lstm_{x}.pth')
            scaler_path = os.path.join(models_dir, f'dengue_scalers_{x}.pkl')
            model.load_model(model_path, scaler_path)
            result_tuple = model.create_X_final()
            df_all = model.transform_X_final(result_tuple)
            df_all.insert(0, 'city', city_names[x])
            dfs[city_names[x]] = df_all

            #Release memory
            del model
            import gc
            gc.collect()
        return dfs