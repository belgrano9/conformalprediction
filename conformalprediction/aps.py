import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.decomposition import PCA


class AdaptivePredictionSet(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, alpha=0.1, test_size=0.2, random_state=None):
        self.base_model = base_model
        self.alpha = alpha
        self.test_size = test_size
        self.random_state = random_state
        self.qhat = None

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Split the data into training and calibration sets
        X_train, X_calib, y_train, y_calib = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Fit the base model
        self.base_model.fit(X_train, y_train)

        # Compute conformity scores on the calibration set
        calib_probs = self.base_model.predict_proba(X_calib)
        conf_scores = self._compute_conformity_scores(calib_probs, y_calib)

        # Compute the quantile
        n = len(y_calib)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.qhat = np.quantile(conf_scores, q_level, method="higher")

        return self

    def _compute_conformity_scores(self, probs, y):
        sorted_probs = np.sort(probs, axis=1)[
            :, ::-1
        ]  # Sort probabilities in descending order
        cumsum_probs = np.cumsum(sorted_probs, axis=1)
        y_ranks = np.argmax(
            probs == np.expand_dims(probs[np.arange(len(y)), y], axis=1), axis=1
        )
        return cumsum_probs[np.arange(len(y)), y_ranks]

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        probs = self.base_model.predict_proba(X)
        pred_sets = self._construct_prediction_sets(probs)

        return pred_sets

    def _construct_prediction_sets(self, probs):
        sorted_indices = np.argsort(probs, axis=1)[:, ::-1]
        cumsum_probs = np.cumsum(
            probs[np.arange(probs.shape[0])[:, np.newaxis], sorted_indices], axis=1
        )

        pred_sets = cumsum_probs < self.qhat

        # Include the label whose cumulative score is just above the quantile
        above_quantile = np.argmax(cumsum_probs >= self.qhat, axis=1)
        pred_sets[np.arange(pred_sets.shape[0]), above_quantile] = True

        # Convert to original label order
        return pred_sets[
            np.arange(pred_sets.shape[0])[:, np.newaxis], np.argsort(sorted_indices)
        ]

    def predict_proba(self, X):
        check_is_fitted(self)
        return self.base_model.predict_proba(X)

    def score(self, X, y):
        pred_sets = self.predict(X)
        return np.mean(pred_sets[np.arange(len(y)), y])

    def visualize_data(self, X, y, show_decision_boundary=False, figsize=(10, 8)):
        """
        Visualize the data points colored by their labels and optionally show the decision boundary.
        If X has more than 2 features, PCA is used to reduce it to 2 dimensions.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The input samples.
        - y: array-like of shape (n_samples,)
            The target values (class labels).
        - show_decision_boundary: bool, optional (default=False)
            Whether to show the decision boundary.
        - figsize: tuple, optional
            Figure size (width, height) in inches.

        Returns:
        - fig: matplotlib.figure.Figure
            The figure object containing the plot.
        - ax: matplotlib.axes.Axes
            The axes object containing the plot.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # If X has more than 2 features, use PCA for visualization
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(X)
            ax.set_xlabel("First Principal Component")
            ax.set_ylabel("Second Principal Component")
        else:
            X_plot = X
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")

        # Scatter plot of the data points
        scatter = ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap="viridis", alpha=0.7)

        ax.set_title("Data Visualization")

        # Add a color bar
        plt.colorbar(scatter)

        if show_decision_boundary and hasattr(self.base_model, "predict"):
            # Create a mesh grid
            x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
            y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)
            )

            # If we used PCA, we need to transform the mesh grid points back to the original space
            if X.shape[1] > 2:
                xy = np.c_[xx.ravel(), yy.ravel()]
                X_mesh = pca.inverse_transform(xy)
            else:
                X_mesh = np.c_[xx.ravel(), yy.ravel()]

            # Predict for each point in the mesh
            Z = self.base_model.predict(X_mesh)
            Z = Z.reshape(xx.shape)

            # Plot the decision boundary
            ax.contourf(xx, yy, Z, alpha=0.4, cmap="viridis")

        plt.tight_layout()
        return fig, ax

    def plot_set_size_histogram(self, X, bins=None, figsize=(10, 6)):
        """
        Plot a histogram of prediction set sizes for the given data.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The input samples to predict on.
        - bins: int or sequence, optional
            Number of histogram bins or bin edges.
        - figsize: tuple, optional
            Figure size (width, height) in inches.

        Returns:
        - fig: matplotlib.figure.Figure
            The figure object containing the plot.
        - ax: matplotlib.axes.Axes
            The axes object containing the plot.
        """
        pred_sets = self.predict(X)
        set_sizes = pred_sets.sum(axis=1)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot bars centered on their values
        unique_sizes, counts = np.unique(set_sizes, return_counts=True)
        bar_width = 0.8
        ax.bar(unique_sizes, counts, width=bar_width, edgecolor="black", align="center")

        # Set x-axis ticks to be the unique set sizes
        ax.set_xticks(unique_sizes)

        # Extend x-axis slightly for better visibility
        ax.set_xlim(min(unique_sizes) - 0.6, max(unique_sizes) + 0.6)

        ax.set_xlabel("Prediction Set Size")
        ax.set_ylabel("Frequency")
        ax.set_title("Histogram of Prediction Set Sizes")

        # Add text with mean and std of set sizes
        mean_size = np.mean(set_sizes)
        std_size = np.std(set_sizes)
        ax.text(
            0.95,
            0.95,
            f"Mean: {mean_size:.2f}\nStd: {std_size:.2f}",
            verticalalignment="top",
            horizontalalignment="right",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        # Add value labels on top of each bar
        for size, count in zip(unique_sizes, counts):
            ax.text(size, count, str(count), ha="center", va="bottom")

        plt.tight_layout()
        return fig, ax

    def get_uncertain_predictions(self, X):
        """
        Retrieve points whose prediction set size is greater than 1.

        Parameters:
        - X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        - uncertain_df: pandas.DataFrame
            A DataFrame containing the uncertain predictions, their set sizes,
            and the original features.
        """
        check_is_fitted(self)

        # Make predictions
        pred_sets = self.predict(X)
        set_sizes = pred_sets.sum(axis=1)

        # Identify uncertain predictions (set size > 1)
        uncertain_mask = set_sizes > 1
        uncertain_X = X[uncertain_mask]
        uncertain_set_sizes = set_sizes[uncertain_mask]

        # Create DataFrame
        uncertain_df = pd.DataFrame(
            uncertain_X, columns=[f"feature_{i}" for i in range(X.shape[1])]
        )
        uncertain_df["set_size"] = uncertain_set_sizes

        # Add probabilities for each class
        probs = self.predict_proba(uncertain_X)
        for i in range(probs.shape[1]):
            uncertain_df[f"prob_class_{i}"] = probs[:, i]

        return uncertain_df
