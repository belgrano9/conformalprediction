import matplotlib.pyplot as plt
import numpy as np

# Define the colors dictionary for plotting
colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}


def plot_scores(alphas, scores, quantiles, method, ax):
    ax.hist(scores, bins="auto")
    for i, quantile in enumerate(quantiles):
        ax.vlines(
            x=quantile,
            ymin=0,
            ymax=100,
            color=colors[i],
            linestyles="dashed",
            label=f"alpha = {alphas[i]}",
        )
    ax.set_title(f"Distribution of scores for '{method}' method")
    ax.legend()
    ax.set_xlabel("scores")
    ax.set_ylabel("count")


def plot_prediction_decision(X_train, y_train, X_test, y_pred_mapie, ax):
    y_train_col = [colors.get(y) for y in y_train]
    y_pred_col = [colors.get(y) for y in y_pred_mapie]
    ax.scatter(
        X_test[:, 0], X_test[:, 1], color=y_pred_col, marker=".", s=10, alpha=0.4
    )
    ax.scatter(
        X_train[:, 0], X_train[:, 1], color=y_train_col, marker="o", s=10, edgecolor="k"
    )
    ax.set_title("Predicted labels")


def plot_prediction_set(X_train, y_train, X_test, y_ps, alpha_, ax):
    tab10 = plt.cm.get_cmap("Purples", 4)
    y_train_col = [colors.get(y) for y in y_train]
    y_pi_sums = y_ps.sum(axis=1)
    num_labels = ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_pi_sums,
        marker="o",
        s=10,
        alpha=1,
        cmap=tab10,
        vmin=0,
        vmax=3,
    )
    ax.scatter(
        X_train[:, 0], X_train[:, 1], color=y_train_col, marker="o", s=10, edgecolor="k"
    )
    ax.set_title(f"Number of labels for alpha={alpha_}")
    plt.colorbar(num_labels, ax=ax)


def plot_results(X_train, y_train, X_test, alphas, y_pred_mapie, y_ps_mapie):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plot_prediction_decision(X_train, y_train, X_test, y_pred_mapie, axes[0, 0])
    for i, alpha_ in enumerate(alphas):
        plot_prediction_set(
            X_train,
            y_train,
            X_test,
            y_ps_mapie[:, :, i],
            alpha_,
            axes[(i + 1) // 2, (i + 1) % 2],
        )
    plt.show()
