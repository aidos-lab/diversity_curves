from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.cluster import AgglomerativeClustering, k_means
import numpy as np


def compute_clustering_scores(X, labels):
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    print("Silhouette Score:", silhouette)
    print("Davies-Bouldin Score:", davies_bouldin)
    print("Calinski-Harabasz Score:", calinski_harabasz)
    return silhouette, davies_bouldin, calinski_harabasz


def compute_rand_kmeans(X, labels_true):
    aris = []
    n_resamples = 100
    for i in range(n_resamples):
        np.random.seed(i)
        indices = np.random.choice(len(X), size=int(len(X) * 0.5), replace=False)
        X_sample = X[indices]
        labels_sample = np.array(labels_true)[indices]
        # using k-means for clustering
        _, label_kmean, _ = k_means(
            X_sample,
            n_clusters=len(np.unique(labels_sample)),
            init="k-means++",
            n_init="auto",
        )
        ari_kmeans = adjusted_rand_score(
            labels_true=labels_sample, labels_pred=label_kmean
        )
        aris.append(ari_kmeans)
    return np.mean(aris), np.std(aris)


def compute_rand_agglo(
    X,
    labels_true,
    metric="euclidean",
):
    aris = []
    n_resamples = 100
    for i in range(n_resamples):
        np.random.seed(i)
        indices = np.random.choice(len(X), size=int(len(X) * 0.5), replace=False)
        if metric == "precomputed":
            rows_sample = X[indices, :]
            X_sample = rows_sample[:, indices]
        else:
            X_sample = X[indices]
        labels_sample = np.array(labels_true)[indices]

        # using agglomerative clustering
        clustering = AgglomerativeClustering(
            n_clusters=len(np.unique(labels_sample)), metric=metric, linkage="average"
        )

        clustering.fit(X_sample)
        labels = clustering.labels_
        aris.append(adjusted_rand_score(labels_true=labels_sample, labels_pred=labels))
    return np.mean(aris), np.std(aris)


def compute_clustering_scores_resample(X, labels):
    silhouette_scores = []
    davies_bouldin_scores = []
    calinski_harabasz_scores = []
    n_resamples = 100
    for i in range(n_resamples):
        np.random.seed(i)  # for reproducibility
        indices = np.random.choice(len(X), size=int(len(X) * 0.5), replace=False)
        X_sample = X[indices]
        labels_sample = np.array(labels)[indices]
        silhouette_scores.append(silhouette_score(X_sample, labels_sample))
        davies_bouldin_scores.append(davies_bouldin_score(X_sample, labels_sample))
        calinski_harabasz_scores.append(
            calinski_harabasz_score(X_sample, labels_sample)
        )

    print(
        "Silhouette Score: {:.3f} ± {:.3f}".format(
            np.mean(silhouette_scores), np.std(silhouette_scores)
        )
    )
    print(
        "Davies-Bouldin Score: {:.3f} ± {:.3f}".format(
            np.mean(davies_bouldin_scores), np.std(davies_bouldin_scores)
        )
    )
    print(
        "Calinski-Harabasz Score: {:.3f} ± {:.3f}".format(
            np.mean(calinski_harabasz_scores), np.std(calinski_harabasz_scores)
        )
    )

    return (
        np.mean(silhouette_scores),
        np.mean(davies_bouldin_scores),
        np.mean(calinski_harabasz_scores),
    ), (
        np.std(silhouette_scores),
        np.std(davies_bouldin_scores),
        np.std(calinski_harabasz_scores),
    )


def compute_silhouette_score_resamples(X, labels, n_resamples=100):
    silhouette_scores = []
    for i in range(n_resamples):
        np.random.seed(i)  # for reproducibility
        indices = np.random.choice(len(X), size=int(len(X) * 0.5), replace=False)
        X_sample = X[indices]
        X_sample = X_sample[:, indices]
        labels_sample = np.array(labels)[indices]
        silhouette_scores.append(
            silhouette_score(X_sample, labels_sample, metric="precomputed")
        )

    print(
        "Silhouette Score: {:.3f} ± {:.3f}".format(
            np.mean(silhouette_scores), np.std(silhouette_scores)
        )
    )
    return np.mean(silhouette_scores), np.std(silhouette_scores)


def all_clustering_results_euclidean(method, df, X, y):
    ari_kmeans_mean, ari_kmeans_std = compute_rand_kmeans(X, y)
    ari_agglo_mean, ari_agglo_std = compute_rand_agglo(X, y, metric="euclidean")

    means, stds = compute_clustering_scores_resample(X, y)

    df[method]["silh_mean"], df[method]["db_mean"], df[method]["ch_mean"] = means
    df[method]["silh_std"], df[method]["db_std"], df[method]["ch_std"] = stds
    df[method]["ari_agglo_mean"] = ari_agglo_mean
    df[method]["ari_agglo_std"] = ari_agglo_std
    df[method]["ari_kmeans_mean"] = ari_kmeans_mean
    df[method]["ari_kmeans_std"] = ari_kmeans_std

    return
