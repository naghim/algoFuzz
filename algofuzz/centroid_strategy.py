from numpy.typing import NDArray
from algofuzz.enums import CentroidStrategy
import numpy as np

from algofuzz.exceptions import AlgofuzzException

def create_centroids(X: NDArray, strategy: CentroidStrategy, num_clusters: int) -> NDArray:
    x_size = X.shape[0]
    y_size = X.shape[1]

    if strategy == CentroidStrategy.Random:
        centroids = np.random.rand(x_size, num_clusters)
        return centroids

    if strategy == CentroidStrategy.Outliers:
        centroids = (np.random.rand(x_size, num_clusters) * 10) + 10
        return centroids

    if strategy == CentroidStrategy.Diagonal:
        centroids = np.column_stack(
            (
                (np.min(X, axis=1) + np.max(X, axis=1)) / 2,
                np.max(X, axis=1),
                np.min(X, axis=1)
            )
        )

        return centroids

    if strategy == CentroidStrategy.Mirtill:
        centroids = np.zeros((X.shape[0], num_clusters))

        for d in range(num_clusters):
            val = d / (num_clusters - 1)
            centroids[:, d] = val

        return centroids

    if strategy == CentroidStrategy.NormalizedIrisDiagonal:
        # TODO: revise this to be generalized, not dataset specific
        centroids = np.array([[(1-(-1)**d)/2, 0.5, (1+(-1)**d)/2] for d in range(x_size)])
        return centroids

    if strategy == CentroidStrategy.NormalizedBreastDiagonal:
        # TODO: revise this to be generalized, not dataset specific
        centroids = np.array([[(1-(-1)**d)/2, (1+(-1)**d)/2] for d in range(x_size)])
        return centroids

    if strategy == CentroidStrategy.Sample:
        random_indices = np.random.choice(y_size, num_clusters, replace=False)
        centroids = X[:,random_indices]
        return centroids

    if strategy == CentroidStrategy.Custom:
        raise AlgofuzzException("Centroids must be set through the <centroids> field.")
