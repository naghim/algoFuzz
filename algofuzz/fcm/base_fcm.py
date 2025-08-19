"""
This module contains the base class for all fuzzy c-means (FCM) implementations.

Inheriting from this class provides default implementations of:

- Evaluation metrics: Purity, Adjusted Rand Index, and Normalized Mutual Information
- Plotting of clusters
- Centroid initialization
- Checking if the model is trained
- Getting information such as the membership matrix, cluster labes, and cluster eta values
"""

from algofuzz import centroid_strategy
from algofuzz.evaluate import evaluate_true_labels
from algofuzz.enums import CentroidStrategy
from algofuzz.exceptions import NotTrainedException
from typing import Optional
from pydantic import BaseModel, Extra, Field
from numpy.typing import ArrayLike, NDArray
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

    def evaluate_true_labels(self, true_labels: NDArray) -> list[float]:
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

        return evaluate_true_labels(self.labels, true_labels)

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

        self.centroids = centroid_strategy.create_centroids(X, self.centroid_strategy, self.num_clusters)

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
