import shap
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

class ShapAnalyzer:
    def __init__(self, model, data, feature_names=None):
        """
        Initialize the ShapAnalyzer class.

        :param model: The prediction model (e.g., Linear Regression, XGBoost, Neural Network).
        :param data: The dataset used for computing SHAP values.
        :param feature_names: Optional list of feature names for better interpretability.
        """
        self.model = model
        self.data = data
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        self.predictions = None

    def initialize_explainer(self):
        """
        Initialize the SHAP explainer based on the type of model provided.
        """
        if hasattr(self.model, 'coef_'):  # Linear regression or similar
            self.explainer = shap.LinearExplainer(self.model, self.data)
        elif hasattr(self.model, 'predict_proba'):  # Neural Networks and Tree-based models
            self.explainer = shap.TreeExplainer(self.model)
        else:
            raise ValueError("Not a listed model.")

    def compute_shap_values(self):
        """
        Compute SHAP values for the provided dataset.
        """
        if self.explainer is None:
            self.initialize_explainer()
        self.shap_values = self.explainer.shap_values(self.data)

    def compute_predictions(self):
        """
        Compute predictions using the model.
        """
        self.predictions = self.model.predict(self.data)

    def compute_prediction_statistics(self):
        """
        Compute summary statistics for the predictions.

        :return: A dictionary containing mean, median, std, lower quantile, upper quantile, skewness, and kurtosis.
        """
        if self.predictions is None:
            self.compute_predictions()

        stats = {
            "mean": np.mean(self.predictions),
            "median": np.median(self.predictions),
            "std": np.std(self.predictions),
            "lower_quantile": np.percentile(self.predictions, 25),
            "upper_quantile": np.percentile(self.predictions, 75),
            "skewness": skew(self.predictions),
            "kurtosis": kurtosis(self.predictions)
        }
        return stats

    def classify_distribution(self, stats):
        """
        Classify the distribution based on skewness and kurtosis.

        :param stats: Dictionary containing skewness and kurtosis.
        :return: A string describing the distribution type.
        """
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
        """
        Analyze bias in the predictions based on mean and median.

        :param stats: Dictionary containing mean and median.
        :return: A string describing potential bias.
        """
        mean = stats["mean"]
        median = stats["median"]

        if abs(mean - median) < 0.1 * mean:
            return "The predictions appear unbiased."
        else:
            return "The predictions may have bias, as the mean and median differ significantly."

    def compute_average_parameters(self):
        """
        Compute the average SHAP values for each feature.

        :return: A dictionary with feature names and their average SHAP values.
        """
        if self.shap_values is None:
            self.compute_shap_values()
        average_shap_values = self.shap_values.mean(axis=0)
        return {name: avg for name, avg in zip(self.feature_names, average_shap_values)}

    def plot_dependence(self, feature_index):
        """
        Generate a SHAP dependence plot for a specific feature.

        :param feature_index: Index of the feature to plot.
        """
        if self.shap_values is None:
            self.compute_shap_values()
        shap.dependence_plot(feature_index, self.shap_values, self.data, feature_names=self.feature_names)

    def plot_mean_shap(self):
        """
        Generate a bar plot showing the mean SHAP values for all features.
        """
        if self.shap_values is None:
            self.compute_shap_values()
        shap.summary_plot(self.shap_values, self.data, feature_names=self.feature_names, plot_type="bar")

    def plot_swarm(self):
        """
        Generate a SHAP swarm plot for all features.
        """
        if self.shap_values is None:
            self.compute_shap_values()
        shap.summary_plot(self.shap_values, self.data, feature_names=self.feature_names)

    def get_most_important_feature(self):
        """
        Identify the most important feature based on SHAP values.

        :return: A tuple containing the feature name and its average SHAP value.
        """
        if self.shap_values is None:
            self.compute_shap_values()
        average_shap_values = np.mean(self.shap_values, axis=0)
        max_index = np.argmax(average_shap_values)
        return self.feature_names[max_index], average_shap_values[max_index]

    def generate_text_summary(self):
        """
        Generate a textual summary of SHAP analysis results and prediction statistics.

        :return: A string containing the summary.
        """
        if self.shap_values is None:
            self.compute_shap_values()

        # Compute average SHAP values
        avg_params = self.compute_average_parameters()

        # Identify the most important feature
        most_important_feature, max_shap_value = self.get_most_important_feature()

        # Compute prediction statistics
        stats = self.compute_prediction_statistics()
        distribution_classification = self.classify_distribution(stats)
        bias_analysis = self.analyze_bias(stats)

        # Generate textual summary
        summary = f"SHAP Analysis Summary:\n"
        summary += f"The most important feature is '{most_important_feature}' with an average SHAP value of {max_shap_value:.2f}.\n"
        summary += f"Here are the average SHAP values for all features:\n"
        for feature, value in avg_params.items():
            summary += f"  - {feature}: {value:.2f}\n"

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
        summary += "  - Mean SHAP plot shows the average impact of each feature.\n"
        summary += "  - Swarm plot visualizes the distribution of SHAP values across all samples.\n"
        summary += "  - Dependence plot highlights the interaction of SHAP values for specific features.\n"

        return summary