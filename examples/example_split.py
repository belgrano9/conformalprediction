import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from conformalprediction.split import SplitCP


import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a base model (Logistic Regression in this case)
base_model = LogisticRegression(random_state=42)

# Create the SplitCP model
cp_model = SplitCP(base_model=base_model, alpha=0.05, test_size=0.2, random_state=42)

# Fit the model
cp_model.fit(X_train, y_train)

# Make predictions
y_pred_sets = cp_model.predict(X_test)

# Calculate and print metrics
average_set_size = y_pred_sets.sum(axis=1).mean()
empirical_coverage = (y_pred_sets[np.arange(len(y_test)), y_test]).mean()
base_model_predictions = cp_model.base_model.predict(X_test)
base_model_accuracy = accuracy_score(y_test, base_model_predictions)

print(f"Quantile value: {cp_model.qhat: .4}")
print(f"Average prediction set size: {average_set_size:.2f}")
print(f"Empirical coverage: {empirical_coverage:.2f}")
print(f"Base model accuracy: {base_model_accuracy:.2f}")

# Visualize the data and decision boundary
# fig, ax = cp_model.visualize_data(X_test, y_test, show_decision_boundary=True)
# plt.show()

# Plot the histogram of prediction set sizes
# fig, ax = cp_model.plot_set_size_histogram(X_test)
# plt.show()
# Get uncertain predictions
uncertain_df = cp_model.get_uncertain_predictions(X_test)
print("\nUncertain predictions:")
print(uncertain_df.head())
print(f"\nNumber of uncertain predictions: {len(uncertain_df)}")
