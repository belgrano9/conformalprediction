import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from conformalprediction.aps import AdaptivePredictionSet
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a base model (Logistic Regression in this case)
base_model = DecisionTreeClassifier(random_state=42)

# Create the APS model
aps_model = AdaptivePredictionSet(
    base_model=base_model, alpha=0.05, test_size=0.2, random_state=42
)

# Fit the model
aps_model.fit(X_train, y_train)

# Make predictions
y_pred_sets = aps_model.predict(X_test)

# Calculate and print metrics
average_set_size = y_pred_sets.sum(axis=1).mean()
empirical_coverage = aps_model.score(X_test, y_test)
base_model_accuracy = base_model.score(X_test, y_test)

print(f"Average prediction set size: {average_set_size:.2f}")
print(f"Empirical coverage: {empirical_coverage:.2f}")
print(f"Base model accuracy: {base_model_accuracy:.2f}")

# Visualize the data and decision boundary
fig, ax = aps_model.visualize_data(X, y, show_decision_boundary=True)
plt.show()

# Plot the histogram of prediction set sizes
fig, ax = aps_model.plot_set_size_histogram(X_test)
plt.show()

# Get uncertain predictions
uncertain_df = aps_model.get_uncertain_predictions(X_test)
print("\nUncertain predictions:")
print(uncertain_df.head())
print(f"\nNumber of uncertain predictions: {len(uncertain_df)}")
