"""
This module contains the implementation of the ST-PFCM algorithm, which is a self-tuning version of the Possibilistic Fuzzy C-Means Clustering algorithm proposed by MB. Naghi in 2023.
"""


from numpy.typing import NDArray
from pydantic import Field
from algofuzz.fcm.base_fcm import BaseFCM
from algofuzz.exceptions import NotTrainedException
import numpy as np
from numba import njit


class STPFCM(BaseFCM):
    """
    Partitions a numeric dataset using the Self-Tuning Possibilistic Fuzzy C-Means Clustering (ST-PFCM) algorithm.
    """
    p: float = Field(default=2.0, gt=1.0)  # Exponent
    """
    The fuzzy exponent parameter. The default value is 2.0. Must be greater than 1.
    """

    kappa: float = Field(default=1, ge=1e-9)  # Kappa
    """
    The penalty factor for the noise cluster. The default value is 1. Must be greater than or equal to 1e-9.
    """

    w_prob: float = Field(default=1.0, gt=0.0)  # a
    """
    Balancing factor, controls the influence of the probabilistic membership u in the clustering process. A higher weight increases the importance of u in updating the centroids.

    The default value is 1.0. Must be greater than 0.
    """

    def fit(self, X: NDArray) -> None:
        """
        Fits the model to the data.

        Parameters:
            X (NDArray): The input data of shape (n_samples, n_features)
        Returns:
            None
        """

        n = X.shape[0]
        z = X.shape[1]

        u = np.zeros((self.num_clusters, n))
        t = np.zeros((self.num_clusters, n))

        corrected_p = 1 / (self.p - 1)
        corrected_m = -2 / (self.m - 1)

        center = np.mean(X, axis=0)

        eta_sum = np.linalg.norm(X[:, :] - center) ** 2  # np.sum()

        eta = (self.kappa / n) * eta_sum

        eta2 = eta
        alpha = np.full(self.num_clusters, 1.0 / self.num_clusters)

        self._create_centroids(X, transpose=False)

        for _ in range(self.max_iter):
            distances = np.linalg.norm(
                X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
            exponents = (
                (distances ** 2) / (eta2 * (alpha[np.newaxis, :] ** (self.m - 1)))) ** corrected_p
            t = 1 / (1 + exponents)
            exact_matches = np.isclose(distances, 0, atol=1e-7)
            u = np.zeros((n, self.num_clusters))

            if np.any(exact_matches):
                exact_indices = np.argmax(exact_matches, axis=1)
                u[np.arange(n), exact_indices] = 1
            non_exact_indices = np.where(~np.any(exact_matches, axis=1))[0]

            if non_exact_indices.size > 0:
                norm_diff_non_exact = distances[non_exact_indices, :]
                u[non_exact_indices, :] = alpha * \
                    (norm_diff_non_exact ** corrected_m)
                u[non_exact_indices, :] /= np.sum(
                    u[non_exact_indices, :], axis=1, keepdims=True)

            alpha = np.sum((self.w_prob * u ** self.m + t ** self.p)
                           * (distances ** 2), axis=0) ** (1 / self.m)
            alpha /= np.sum(alpha)

            sumcur = (self.w_prob * u **
                      self.m + t ** self.p)

            weighted_sum = np.dot(sumcur.T, X)

            sumdn = np.sum(sumcur, axis=0)

            self.centroids = np.divide(weighted_sum, sumdn[:, np.newaxis], out=np.zeros_like(
                weighted_sum), where=sumdn[:, np.newaxis] != 0)

        self._eta = eta2 * (alpha ** (self.m - 1))
        self._member = self.w_prob * u ** self.m + t ** self.p
        self._alpha = alpha
        self.trained = True
