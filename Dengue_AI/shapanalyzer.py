import shap
import torch
import matplotlib.pyplot as plt
from model_loader import DengueLSTM
import numpy as np
import os

class ShapAnalyzer:
    def __init__(self, model, features_df, city=None, sample_size=1000):
        self.city = city
        self.sample_size = sample_size
        self.model_obj = DengueLSTM(city=city)
        
        model_path = os.path.join("Models", f'dengue_lstm_{city}.pth')
        scaler_path = os.path.join("Models", f'dengue_scalers_{city}.pkl')
        
        self.model_obj.load_model(model_path, scaler_path)
        self.model = self.model_obj.model
        self.X_train = self.model_obj.X_train_torch 
        self.shap_values_2d = None
        self.X_train_2d = None
        self.feature_names = None
        self.predictions = None

    def compute_shap(self):
        background = self.X_train[:min(5, self.X_train.shape[0])]
        explainer = shap.DeepExplainer(self.model, background)
        try:
            shap_values = explainer.shap_values(self.X_train[:self.sample_size])
        except AssertionError as e:
            print("Warning: SHAP DeepExplainer assertion error encountered.")
            print(str(e))
            print("This is likely due to unsupported operations in your LSTM model for SHAP DeepExplainer.")
            print("You can ignore this error for visualization purposes, but SHAP values may not sum to the model output.")
            try:
                shap_values = explainer.shap_values(self.X_train[:self.sample_size], check_additivity=False)
            except Exception as e2:
                print("Failed to compute SHAP values even with check_additivity=False.")
                raise e2
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shape = shap_values.shape
        num_samples = shape[0]
        feature_dim = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        self.shap_values_2d = shap_values.reshape(num_samples, feature_dim)
        self.X_train_2d = self.X_train[:self.sample_size].numpy().reshape(num_samples, feature_dim)
        # Generate default feature names if not provided
        if hasattr(self.model_obj, "sj_top_features") and self.model_obj.sj_top_features:
            seq_len = self.X_train.shape[1]
            base_features = self.model_obj.sj_top_features
            self.feature_names = [f"{feat}_t{t}" for t in range(seq_len) for feat in base_features]
        else:
            self.feature_names = [f"feature_{i}" for i in range(feature_dim)]

    def plot(self):
        if self.shap_values_2d is None or self.X_train_2d is None:
            self.compute_shap()
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values_2d, self.X_train_2d, feature_names=self.feature_names, show=True)

    def plot_mean_shap(self):
        if self.shap_values_2d is None or self.X_train_2d is None:
            self.compute_shap()
        shap.summary_plot(self.shap_values_2d, self.X_train_2d, feature_names=self.feature_names, plot_type="bar", show=True)

    def plot_swarm(self):
        if self.shap_values_2d is None or self.X_train_2d is None:
            self.compute_shap()
        shap.summary_plot(self.shap_values_2d, self.X_train_2d, feature_names=self.feature_names, show=True)

    def plot_dependence(self, feature_index):
        if self.shap_values_2d is None or self.X_train_2d is None:
            self.compute_shap()
        shap.dependence_plot(feature_index, self.shap_values_2d, self.X_train_2d, feature_names=self.feature_names, show=True)

    def compute_predictions(self):
        self.model.eval()
        with torch.no_grad():
            X_tensor = self.X_train[:self.sample_size]
            output = self.model(X_tensor)
            if isinstance(output, tuple):
                output = output[0]
            output_np = output.cpu().numpy()
            if output_np.ndim == 0:
                self.predictions = np.array([output_np.item()])
            else:
                self.predictions = output_np.squeeze()

    def compute_prediction_statistics(self):
        if self.predictions is None:
            self.compute_predictions()
        stats = {
            "mean": np.mean(self.predictions),
            "median": np.median(self.predictions),
            "std": np.std(self.predictions),
            "lower_quantile": np.percentile(self.predictions, 25),
            "upper_quantile": np.percentile(self.predictions, 75),
            "skewness": (self.predictions - np.mean(self.predictions)).sum() / len(self.predictions) if len(self.predictions) > 0 else 0,
            "kurtosis": ((self.predictions - np.mean(self.predictions))**4).sum() / len(self.predictions) if len(self.predictions) > 0 else 0
        }
        return stats

    def classify_distribution(self, stats):
        skewness = stats["skewness"]
        kurt = stats["kurtosis"]
        if abs(skewness) < 0.5 and kurt < 3:
            return "The distribution is approximately normal."
        elif skewness > 0.5:
            return "The distribution is positively skewed."
        elif skewness < -0.5:
            return "The distribution is negatively skewed."
        else:
            return "The distribution has heavy tails."

    def analyze_bias(self, stats):
        mean = stats["mean"]
        median = stats["median"]
        if abs(mean - median) < 0.1 * mean:
            return "The predictions appear unbiased."
        else:
            return "The predictions may have bias, as the mean and median differ significantly."

    def compute_average_parameters(self):
        if self.shap_values_2d is None:
            self.compute_shap()
        avg_shap = np.mean(self.shap_values_2d, axis=0)
        return {name: float(val) for name, val in zip(self.feature_names, avg_shap)}

    def get_most_important_feature(self):
        if self.shap_values_2d is None:
            self.compute_shap()
        avg_shap = np.mean(np.abs(self.shap_values_2d), axis=0)
        max_index = np.argmax(avg_shap)
        feature_name = self.feature_names[max_index] if self.feature_names else f"feature_{max_index}"
        return feature_name, float(avg_shap[max_index])

    def generate_text_summary(self):
        if self.shap_values_2d is None:
            self.compute_shap()
       
        most_important_feature, max_shap_value = self.get_most_important_feature()
        stats = self.compute_prediction_statistics()
        distribution_classification = self.classify_distribution(stats)
        bias_analysis = self.analyze_bias(stats)
        summary = f"SHAP Analysis Summary:\n"
        summary += f"The most important feature is '{most_important_feature}' with an average SHAP value of {max_shap_value:.2f}.\n"
        summary += "\nPrediction Statistics:\n"
        summary += f"  - Mean: {stats['mean']:.2f}\n"
        summary += f"  - Median: {stats['median']:.2f}\n"
        summary += f"  - Standard Deviation: {stats['std']:.2f}\n"
        summary += f"  - Lower Quantile (25%): {stats['lower_quantile']:.2f}\n"
        summary += f"  - Upper Quantile (75%): {stats['upper_quantile']:.2f}\n"
        summary += f"  - Skewness: {stats['skewness']:.2f}\n"
        summary += f"  - Kurtosis: {stats['kurtosis']:.2f}\n"
        summary += f"\nDistribution Analysis:\n{distribution_classification}\n"
        summary += f"\nBias Analysis:\n{bias_analysis}\n"
        summary += "\nPlots generated:\n"
        if self.shap_values_2d.size == 1:
            summary += "  - Only one SHAP value available, plots cannot be generated.\n"
        else:
            summary += "  - Mean SHAP plot shows the average impact of each feature.\n"
            summary += "  - Swarm plot visualizes the distribution of SHAP values across all samples.\n"
            summary += "  - Dependence plot highlights the interaction of SHAP values for specific features.\n"
        return summary