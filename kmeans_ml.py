"""Cached K-Means and flight-delay analysis used by the R2A2 Flask module."""

import base64
import io
import os
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_MAIN = os.path.join(SCRIPT_DIR, "data", "flights_dataset.csv")
DATA_MANUAL = os.path.join(SCRIPT_DIR, "data", "flights_manual_100.csv")
FEATURE_COLUMNS = ["distance_km", "duration_min"]
INITIAL_CENTROIDS = np.array(
    [[400.0, 55.0], [1600.0, 165.0], [2700.0, 280.0]], dtype=float
)
PALETTE = {
    1: "#e63946",
    2: "#2a9d8f",
    3: "#457b9d",
}
CLUSTER_NAMES = {
    1: "Short-haul segment",
    2: "Medium-haul segment",
    3: "Long-haul segment",
}


def _read_dataset(path, required_columns, minimum_rows):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")
    if len(frame) < minimum_rows:
        raise ValueError(f"{path} must contain at least {minimum_rows} records")
    for column in required_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy()).all():
            raise ValueError(f"Column {column} contains a non-finite value")
    return frame


@lru_cache(maxsize=1)
def _manual_data():
    return _read_dataset(DATA_MANUAL, ["flight_id"] + FEATURE_COLUMNS, 100)


@lru_cache(maxsize=1)
def _flight_data():
    return _read_dataset(
        DATA_MAIN,
        ["flight_id"] + FEATURE_COLUMNS + ["delayed"],
        1000,
    )


def load_manual():
    return _manual_data().copy(deep=True)


def load_flights():
    return _flight_data().copy(deep=True)


def euclidean(first, second):
    return float(np.sqrt(np.sum((first - second) ** 2)))


def _style_axes(ax, title, xlabel, ylabel):
    ax.set_xlabel(xlabel, color="#e2e8f0", labelpad=10)
    ax.set_ylabel(ylabel, color="#e2e8f0", labelpad=10)
    ax.set_title(title, color="#f1f5f9", fontweight="bold", pad=12)
    ax.set_facecolor("#131b2e")
    ax.tick_params(colors="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    ax.grid(alpha=0.25, linestyle="--")


def _fig_to_b64(fig):
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=100,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _manual_initial_plot(values):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.scatter(
        values[:, 0],
        values[:, 1],
        color="#94a3b8",
        alpha=0.72,
        edgecolors="white",
        linewidths=0.35,
        s=34,
        label="Flight records",
    )
    ax.scatter(
        INITIAL_CENTROIDS[:, 0],
        INITIAL_CENTROIDS[:, 1],
        marker="X",
        s=180,
        color="#f8fafc",
        edgecolors="#4d6bfe",
        linewidths=1.2,
        label="Initial centroids",
    )
    for index, centroid in enumerate(INITIAL_CENTROIDS, start=1):
        ax.annotate(
            f"C{index}",
            centroid,
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            color="#f8fafc",
            fontweight="bold",
        )
    ax.legend(
        loc="upper left",
        framealpha=0.9,
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#e2e8f0",
    )
    _style_axes(
        ax,
        "Manual exercise: 100 flights and initial centroids",
        "Distance (km)",
        "Duration (min)",
    )
    fig.patch.set_facecolor("#0b0f19")
    return _fig_to_b64(fig)


def _manual_iteration_plot(values, assignments, centroids, iteration):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for cluster_id in range(1, 4):
        mask = assignments == cluster_id
        ax.scatter(
            values[mask, 0],
            values[mask, 1],
            color=PALETTE[cluster_id],
            alpha=0.72,
            edgecolors="white",
            linewidths=0.35,
            s=34,
            label=f"Cluster {cluster_id}",
        )
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=190,
        color="#f8fafc",
        edgecolors="#0b0f19",
        linewidths=1.1,
        label="Updated centroids",
        zorder=5,
    )
    for index, centroid in enumerate(centroids, start=1):
        ax.annotate(
            f"C{index}",
            centroid,
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            color="#f8fafc",
            fontweight="bold",
            zorder=6,
        )
    ax.legend(
        loc="upper left",
        framealpha=0.9,
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#e2e8f0",
        ncol=2,
    )
    _style_axes(
        ax,
        f"Manual K-Means iteration {iteration}: assignments and updated centroids",
        "Distance (km)",
        "Duration (min)",
    )
    fig.patch.set_facecolor("#0b0f19")
    return _fig_to_b64(fig)


def _wcss_plot(iterations):
    values = [item["wcss_after"] for item in iterations]
    labels = [f"Iteration {item['iteration']}" for item in iterations]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    bars = ax.bar(labels, values, color=[PALETTE[index] for index in range(1, 4)])
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            color="#f1f5f9",
            fontsize=9,
        )
    _style_axes(
        ax,
        "Within-cluster sum of squares after each centroid update",
        "Iteration",
        "WCSS",
    )
    fig.patch.set_facecolor("#0b0f19")
    return _fig_to_b64(fig)


def _build_manual_view():
    frame = load_manual()
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    centroids = INITIAL_CENTROIDS.copy()
    iterations = []

    for iteration in range(1, 4):
        distances = np.column_stack(
            [np.linalg.norm(values - centroid, axis=1) for centroid in centroids]
        )
        assignments = np.argmin(distances, axis=1) + 1
        new_centroids = np.zeros_like(centroids)
        for cluster_id in range(1, 4):
            mask = assignments == cluster_id
            if np.any(mask):
                new_centroids[cluster_id - 1] = values[mask].mean(axis=0)
            else:
                new_centroids[cluster_id - 1] = centroids[cluster_id - 1]

        wcss_before = float(
            np.sum(
                [
                    distances[row_index, assignments[row_index] - 1] ** 2
                    for row_index in range(len(values))
                ]
            )
        )
        updated_distances = np.linalg.norm(
            values - new_centroids[assignments - 1], axis=1
        )
        wcss_after = float(np.sum(updated_distances ** 2))
        rows = []
        for row_index, record in frame.iterrows():
            row_distances = distances[row_index]
            assigned_cluster = int(assignments[row_index])
            rows.append(
                {
                    "flight_id": int(record["flight_id"]),
                    "distance_km": round(float(record["distance_km"]), 1),
                    "duration_min": round(float(record["duration_min"]), 1),
                    "d1": round(float(row_distances[0]), 2),
                    "d2": round(float(row_distances[1]), 2),
                    "d3": round(float(row_distances[2]), 2),
                    "assigned_cluster": assigned_cluster,
                    "assigned_distance": round(
                        float(row_distances[assigned_cluster - 1]), 2
                    ),
                }
            )
        iterations.append(
            {
                "iteration": iteration,
                "distances": distances,
                "assignments": assignments,
                "centroids_before": centroids,
                "centroids_after": new_centroids,
                "counts": [
                    int(np.sum(assignments == cluster_id)) for cluster_id in range(1, 4)
                ],
                "wcss_before": round(wcss_before, 2),
                "wcss_after": round(wcss_after, 2),
                "wcss": round(wcss_after, 2),
                "within_cluster_variance": round(wcss_after / len(values), 2),
                "rows": rows,
                "plot": _manual_iteration_plot(
                    values, assignments, new_centroids, iteration
                ),
            }
        )
        centroids = new_centroids

    final_order = np.argsort(centroids[:, 0])
    final_profiles = []
    for semantic_id, raw_index in enumerate(final_order, start=1):
        profile_values = values[assignments == raw_index + 1]
        final_profiles.append(
            {
                "id": semantic_id,
                "name": CLUSTER_NAMES[semantic_id],
                "count": int(len(profile_values)),
                "centroid": [
                    round(float(centroids[raw_index, 0]), 2),
                    round(float(centroids[raw_index, 1]), 2),
                ],
                "mean_distance_km": round(float(profile_values[:, 0].mean()), 1),
                "mean_duration_min": round(float(profile_values[:, 1].mean()), 1),
            }
        )

    return {
        "n_rows": len(frame),
        "initial_centroids": [
            [round(float(value), 2) for value in centroid]
            for centroid in INITIAL_CENTROIDS
        ],
        "initial_plot": _manual_initial_plot(values),
        "iterations": iterations,
        "wcss_plot": _wcss_plot(iterations),
        "final_centroids": [
            [round(float(value), 2) for value in centroid] for centroid in centroids
        ],
        "final_profiles": final_profiles,
        "dataset": frame.head(10).to_dict("records"),
    }


@lru_cache(maxsize=1)
def manual_view():
    return _build_manual_view()


def run_manual_simulation():
    return _build_manual_view()


def render_iteration_plots(state):
    return state


def wcss_plot(state):
    if state.get("wcss_plot"):
        return state["wcss_plot"]
    return _wcss_plot(state["iterations"])


def distance_table(state, n_rows=None):
    rows = state["iterations"][0]["rows"]
    return rows if n_rows is None else rows[:n_rows]


def _kmeans_plot(values, labels, centers, silhouette):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for cluster_id in range(1, 4):
        mask = labels == cluster_id
        ax.scatter(
            values[mask, 0],
            values[mask, 1],
            color=PALETTE[cluster_id],
            alpha=0.48,
            edgecolors="white",
            linewidths=0.25,
            s=16,
            label=CLUSTER_NAMES[cluster_id],
        )
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=220,
        color="#f8fafc",
        edgecolors="#0b0f19",
        linewidths=1.2,
        label="Centroids",
        zorder=5,
    )
    for cluster_id, center in enumerate(centers, start=1):
        ax.annotate(
            f"C{cluster_id}",
            center,
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="#f8fafc",
            fontweight="bold",
            zorder=6,
        )
    ax.legend(
        loc="upper left",
        framealpha=0.9,
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#e2e8f0",
    )
    _style_axes(
        ax,
        f"K-Means flight segmentation (silhouette = {silhouette:.3f})",
        "Distance (km)",
        "Duration (min)",
    )
    fig.patch.set_facecolor("#0b0f19")
    return _fig_to_b64(fig)


def _build_application_view():
    frame = load_flights()
    raw_values = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(raw_values)
    model = KMeans(
        n_clusters=3,
        init="k-means++",
        random_state=42,
        n_init=10,
    )
    raw_labels = model.fit_predict(scaled_values)
    raw_centers = scaler.inverse_transform(model.cluster_centers_)
    order = np.argsort(raw_centers[:, 0])
    remap = {int(raw_id): semantic_id for semantic_id, raw_id in enumerate(order, 1)}
    labels = np.array([remap[int(label)] for label in raw_labels], dtype=int)
    centers = raw_centers[order]
    frame["Cluster"] = labels
    silhouette = float(silhouette_score(scaled_values, labels))

    profiles = []
    for cluster_id in range(1, 4):
        mask = labels == cluster_id
        cluster_frame = frame.loc[mask]
        cluster_values = raw_values[mask]
        profiles.append(
            {
                "id": cluster_id,
                "name": CLUSTER_NAMES[cluster_id],
                "count": int(mask.sum()),
                "share": round(float(mask.mean() * 100), 1),
                "centroid": [
                    round(float(centers[cluster_id - 1, 0]), 1),
                    round(float(centers[cluster_id - 1, 1]), 1),
                ],
                "mean_distance_km": round(float(cluster_values[:, 0].mean()), 1),
                "mean_duration_min": round(float(cluster_values[:, 1].mean()), 1),
                "min_distance_km": round(float(cluster_values[:, 0].min()), 1),
                "max_distance_km": round(float(cluster_values[:, 0].max()), 1),
                "min_duration_min": round(float(cluster_values[:, 1].min()), 1),
                "max_duration_min": round(float(cluster_values[:, 1].max()), 1),
                "delay_rate": round(float(cluster_frame["delayed"].mean() * 100), 1),
                "within_variance": round(
                    float(
                        np.mean(
                            np.sum(
                                (cluster_values - centers[cluster_id - 1]) ** 2,
                                axis=1,
                            )
                        )
                    ),
                    2,
                ),
            }
        )

    k_scores = []
    for k_value in range(2, 6):
        candidate = KMeans(
            n_clusters=k_value,
            init="k-means++",
            random_state=42,
            n_init=10,
        ).fit(scaled_values)
        k_scores.append(
            {
                "k": k_value,
                "inertia": round(float(candidate.inertia_), 2),
                "silhouette": round(
                    float(silhouette_score(scaled_values, candidate.labels_)), 4
                ),
            }
        )

    records = frame[
        ["flight_id", "distance_km", "duration_min", "delayed", "Cluster"]
    ].round(2).to_dict("records")
    if silhouette >= 0.5:
        interpretation = "The clusters are well separated in the standardized feature space."
    elif silhouette >= 0.25:
        interpretation = "The clusters have moderate separation and should be interpreted cautiously."
    else:
        interpretation = "The clusters have weak separation; additional features are recommended."

    return {
        "n_records": len(frame),
        "feature_ranges": {
            "distance": [
                round(float(frame["distance_km"].min()), 1),
                round(float(frame["distance_km"].max()), 1),
            ],
            "duration": [
                round(float(frame["duration_min"].min()), 1),
                round(float(frame["duration_min"].max()), 1),
            ],
        },
        "records": records,
        "sample": records[:15],
        "clusters": profiles,
        "summary_count": {profile["id"]: profile["count"] for profile in profiles},
        "centroids_raw": [profile["centroid"] for profile in profiles],
        "k_scores": k_scores,
        "silhouette": round(silhouette, 4),
        "silhouette_interpretation": interpretation,
        "plot": _kmeans_plot(raw_values, labels, centers, silhouette),
    }


@lru_cache(maxsize=1)
def application_view():
    return _build_application_view()


def cluster_application():
    return application_view()


@lru_cache(maxsize=1)
def _fit_delay_model():
    frame = load_flights()
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    target = frame["delayed"].to_numpy(dtype=int)
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("The delayed target must contain both classes 0 and 1")
    train_values, test_values, train_target, test_target = train_test_split(
        values,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    model.fit(train_values, train_target)
    return model, train_values, test_values, train_target, test_target


def _confusion_plot(matrix):
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["On-time (0)", "Delayed (1)"])
    ax.set_yticks([0, 1], ["On-time (0)", "Delayed (1)"])
    ax.set_xlabel("Predicted class", color="#e2e8f0", labelpad=8)
    ax.set_ylabel("Actual class", color="#e2e8f0", labelpad=8)
    ax.set_title(
        "Logistic Regression confusion matrix",
        color="#f1f5f9",
        fontweight="bold",
        pad=12,
    )
    for row_index in range(2):
        for column_index in range(2):
            value = int(matrix[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                value,
                ha="center",
                va="center",
                fontweight="bold",
                color="white" if value > np.max(matrix) / 2 else "#0b0f19",
            )
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.patch.set_facecolor("#0b0f19")
    ax.tick_params(colors="#94a3b8")
    return _fig_to_b64(fig)


@lru_cache(maxsize=1)
def classification_application():
    model, train_values, test_values, train_target, test_target = _fit_delay_model()
    predicted = model.predict(test_values)
    matrix = confusion_matrix(test_target, predicted, labels=[0, 1])
    logistic = model.named_steps["logisticregression"]
    metrics = {
        "accuracy": round(float(accuracy_score(test_target, predicted)), 4),
        "precision": round(
            float(precision_score(test_target, predicted, zero_division=0)), 4
        ),
        "recall": round(float(recall_score(test_target, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(test_target, predicted, zero_division=0)), 4),
    }
    target_counts = {
        "on_time": int((load_flights()["delayed"] == 0).sum()),
        "delayed": int((load_flights()["delayed"] == 1).sum()),
    }
    logistic_result = {
        "name": "Logistic Regression",
        **metrics,
        "coefficients": [round(float(value), 4) for value in logistic.coef_[0]],
        "intercept": round(float(logistic.intercept_[0]), 4),
        "confusion": matrix.tolist(),
    }
    return {
        "model_name": "Logistic Regression",
        "features": FEATURE_COLUMNS,
        "n_train": int(len(train_values)),
        "n_test": int(len(test_values)),
        "target_counts": target_counts,
        "positive_rate": round(float(test_target.mean()), 4),
        "pos_rate": round(float(test_target.mean()), 4),
        "metrics": metrics,
        "coefficients": logistic_result["coefficients"],
        "intercept": logistic_result["intercept"],
        "confusion": matrix.tolist(),
        "confusion_plot": _confusion_plot(matrix),
        "logistic": logistic_result,
        "model": model,
    }


def predict_flight(distance_km, duration_min):
    model, _, _, _, _ = _fit_delay_model()
    values = np.array([[distance_km, duration_min]], dtype=float)
    probability = float(model.predict_proba(values)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "label": "Delayed flight" if prediction else "On-time flight",
    }
