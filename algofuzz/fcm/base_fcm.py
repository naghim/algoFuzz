from algofuzz.validation.confusion_matrix import find_best_permutation
from algofuzz.validation.validity_index import adjusted_rand_index, normalized_mutual_information, purity
from algofuzz.enums import CentroidStrategy
from algofuzz.exceptions import NotTrainedException, AlgofuzzException
from typing import Optional, Literal
from pydantic import BaseModel, Extra, Field
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import random

class BaseFCM(BaseModel):
    """
    Base class of all FCM implementations.
    """

    num_clusters: int = Field(default=5, ge=1)
    max_iter: int = Field(default=150, ge=1)
    m: float = Field(default=2.0, gt=1.0) # Fuzzifier parameter
    centroids: Optional[NDArray] = Field(default=None)
    centroid_strategy: Optional[CentroidStrategy] = Field(default=CentroidStrategy.Random)

    trained: Literal[bool] = False

    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = True

    def fit(self, X: NDArray) -> None:
        raise NotImplementedError()

    def evaluate(self, true_labels: NDArray) -> list[float]:
        if not self.is_trained():
            raise NotTrainedException()

        conf_matrix = confusion_matrix(true_labels, self.labels[:len(true_labels)])
        best_permuted_confusion = find_best_permutation(conf_matrix)

        pur = purity(best_permuted_confusion)
        ari = adjusted_rand_index(best_permuted_confusion)
        nmi = normalized_mutual_information(best_permuted_confusion)

        return pur, ari, nmi

    def is_trained(self) -> bool:
        return self.trained

    def plot_clusters(self, X: NDArray) -> None:
        if not self.is_trained():
            raise NotTrainedException()

        labels = self.labels
        rand_gen = random.Random()
        rand_gen.seed(0)

        plt.figure(figsize=(8, 6))

        for i in range(self.num_clusters):
            cluster_points = X[:, labels == i]

            plt.scatter(cluster_points[0], cluster_points[1], c=self._random_color(rand_gen), label=f'Cluster {i+1}')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Clusters')
        plt.legend()
        plt.axis('equal')
        plt.show()

    def _random_color(self, rand_gen: random.Random = None) -> NDArray:
        if rand_gen is None:
            rand_gen = random.Random()

        return np.array([[rand_gen.uniform(0, 1), rand_gen.uniform(0, 1), rand_gen.uniform(0, 1)]])

    def _create_centroids(self, X: NDArray) -> NDArray:
        if self.centroids is not None:
            return

        x_size = X.shape[0]
        y_size = X.shape[1]

        if self.centroid_strategy == CentroidStrategy.Random:
            self.centroids = np.random.rand(x_size, self.num_clusters)
            return

        if self.centroid_strategy == CentroidStrategy.Outliers:
            self.centroids = (np.random.rand(x_size, self.num_clusters) * 10) + 10
            return

        if self.centroid_strategy == CentroidStrategy.Diagonal:
            self.centroids = np.column_stack(
                (
                    (np.min(X, axis=1) + np.max(X, axis=1)) / 2,
                    np.max(X, axis=1),
                    np.min(X, axis=1)
                    )
            )
            return

        if self.centroid_strategy == CentroidStrategy.NormalizedIrisDiagonal:
            # TODO: revise this to be generalized, not dataset specific
            self.centroids = np.array([[(1-(-1)**d)/2, 0.5, (1+(-1)**d)/2] for d in range(x_size)])
            return

        if self.centroid_strategy == CentroidStrategy.NormalizedBreastDiagonal:
            # TODO: revise this to be generalized, not dataset specific
            self.centroids = np.array([[(1-(-1)**d)/2, (1+(-1)**d)/2] for d in range(x_size)])
            return

        if self.centroid_strategy == CentroidStrategy.Sample:
            random_indices = np.random.choice(y_size, self.num_clusters, replace=False)
            self.centroids = X[:,random_indices]
            return

        if self.centroid_strategy == CentroidStrategy.Custom:
            raise AlgofuzzException("Centroids must be set through the <centroids> field.")

    @property
    def member(self) -> NDArray:
        if not self.is_trained():
            raise NotTrainedException()

        return self._member

    @property
    def labels(self) -> NDArray:
        if not self.is_trained():
            raise NotTrainedException()

        return np.argmax(self._member, axis=0)

    @property
    def cluster_eta(self) -> NDArray:
        if not self.is_trained():
            raise NotTrainedException()

        return self._eta
