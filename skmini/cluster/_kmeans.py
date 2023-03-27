import numpy as np

from ..base import ClusterMixin
from ..metrics.pairwise import euclidean_distances


class KMeans(ClusterMixin):
    '''K-Means clustering'''
    def __init__(self, init='k-means++', n_clusters=8, max_iter=300,
                 random_state=None, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.init = init

    def fit(self, X):
        self.centroids = self._init_centroids(X)

        for _ in range(self.max_iter):
            clusters = self._create_clusters(X)
            new_centroids = self._get_new_centroids(X, clusters)

            if self._is_converged(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

        self.labels_ = self._get_cluster_labels(X)

    def _init_centroids(self, X):
        rng = np.random.RandomState(self.random_state)

        if self.init == 'random':
            return X[rng.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.init == 'k-means++':
            # KMeans++ implementation
            centroids = np.zeros((self.n_clusters, X.shape[1]))
            centroids[0] = X[rng.randint(X.shape[0])]

            for k in range(1, self.n_clusters):
                max_idx, max_dist = -1, -1
                for idx in range(X.shape[0]):
                    dist = np.min([euclidean_distances(X[idx], point)
                                   for point in centroids[:k]])
                    if dist > max_dist:
                        max_dist = dist
                        max_idx = idx
                centroids[k] = X[max_idx]
            return centroids
        else:
            raise ValueError("init should be either 'k-means++', 'random', "
                             f"got '{self.init}' instead")

    def _create_clusters(self, X):
        clusters = [[] for _ in range(self.n_clusters)]
        for idx, sample in enumerate(X):
            centroid_idx = self._nearest_centroid_idx(sample)
            clusters[centroid_idx].append(idx)
        return clusters

    def _nearest_centroid_idx(self, sample):
        dist = [euclidean_distances(sample, point) for point in self.centroids]
        return np.argmin(dist)

    def _get_new_centroids(self, X, clusters):
        # new_centroids = np.zeros((self.n_clusters, self.n_features))
        new_centroids = np.zeros_like(self.centroids)
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(X[cluster], axis=0)
            new_centroids[cluster_idx] = cluster_mean
        return new_centroids

    def _is_converged(self, centroids_old, centroids):
        dist = [euclidean_distances(centroids_old[i], centroids[i])
                for i in range(self.n_clusters)]
        return sum(dist) <= self.tol

    def _get_cluster_labels(self, X):
        labels = np.zeros(X.shape[0], dtype=np.int)
        for idx, sample in enumerate(X):
            labels[idx] = self._nearest_centroid_idx(sample)
        return labels
