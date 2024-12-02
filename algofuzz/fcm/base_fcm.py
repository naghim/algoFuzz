"""
This module contains the base class for all fuzzy c-means (FCM) implementations.

Inheriting from this class provides default implementations of:

- Evaluation metrics: Purity, Adjusted Rand Index, and Normalized Mutual Information
- Plotting of clusters
- Centroid initialization
- Checking if the model is trained
- Getting information such as the membership matrix, cluster labes, and cluster eta values
"""

from algofuzz.validation.confusion_matrix import find_best_permutation
from algofuzz.validation.validity_index import adjusted_rand_index, normalized_mutual_information, purity
from algofuzz.enums import CentroidStrategy
from algofuzz.exceptions import NotTrainedException, AlgofuzzException
from typing import Optional, Literal
from pydantic import BaseModel, Extra, Field
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import numpy as np
import random

class BaseFCM(BaseModel):
    """
    Base class of all FCM implementations.
    """

    num_clusters: int = Field(default=5, ge=1)
    """
    The number of clusters to form. The default value is 5. Must be greater than 0.
    """


    max_iter: int = Field(default=150, ge=1)
    """
    The maximum number of iterations to perform. The default value is 150. Must be greater than 0.
    """


    m: float = Field(default=2.0, ge=1.0)
    """ 
    The fuzzifier parameter. A value of 1.0 corresponds to hard clustering, while a value greater than 1.0 corresponds to soft clustering. The default value is 2.0. Must be greater than 1.0.
    """

    centroids: Optional[ArrayLike] = Field(default=None)
    """
    The initial centroids of the clusters. If not provided, the centroids will be initialized using the specified strategy.
    """

    centroid_strategy: Optional[CentroidStrategy] = Field(default=CentroidStrategy.Mirtill)
    """
    The strategy to use for initializing the centroids of the clusters. If not provided, the centroids will be initialized randomly.
    """

    trained: bool = False
    """
    A flag indicating whether the model has been trained. The default value is False.
    """

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def fit(self, X: NDArray) -> None:
        """
        Fits the FCM model to the data.

        Parameters:
            X (np.ndarray): The input data.
        Returns:
            None
        """
        raise NotImplementedError()

    def evaluate(self, true_labels: NDArray) -> list[float]:
        """
        Evaluate the clustering results. Currently uses the true labels of the dataset to perform the evaluations.

        Parameters:
            true_labels (np.ndarray): The true labels of the dataset.

        Returns:
            list[float]: A list containing the following evaluation metrics:
                - Purity
                - Adjusted Rand Index
                - Normalized Mutual Information
        """
        if not self.is_trained():
            raise NotTrainedException()

        conf_matrix = confusion_matrix(true_labels, self.labels[:len(true_labels)])
        best_permuted_confusion = find_best_permutation(conf_matrix)

        pur = purity(best_permuted_confusion)
        ari = adjusted_rand_index(best_permuted_confusion)
        nmi = normalized_mutual_information(best_permuted_confusion)

        return pur, ari, nmi

    def is_trained(self) -> bool:
        """
        Check if the model has been trained.

        Parameters:
            None
        Returns:
            bool: True if the model has been trained, False otherwise.
        """
        return self.trained

    def plot_clusters(self, X: NDArray) -> None:
        """
        Plot the clusters in a 2D space.

        Parameters:
            X (np.ndarray): The data points.
        Returns:    
            None
        """
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

        if self.centroid_strategy == CentroidStrategy.Mirtill:
            self.centroids = np.zeros((X.shape[0], self.num_clusters))

            for d in range(self.num_clusters):
                val = d / (self.num_clusters - 1)
                self.centroids[:, d] = val

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
