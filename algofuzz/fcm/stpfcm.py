"""
This module contains the implementation of the ST-PFCM algorithm, which is a self-tuning version of the Possibilistic Fuzzy C-Means Clustering algorithm proposed by MB. Naghi in 2023.
"""


from numpy.typing import NDArray
from pydantic import Field
from algofuzz.fcm.base_fcm import BaseFCM
from algofuzz.exceptions import NotTrainedException
import numpy as np

class STPFCM(BaseFCM):
    """
    Partitions a numeric dataset using the Self-Tuning Possibilistic Fuzzy C-Means Clustering (ST-PFCM) algorithm.
    """
    p: float = Field(default=2.0, gt=1.0) # Exponent
    """
    The fuzzy exponent parameter. The default value is 2.0. Must be greater than 1.
    """

    kappa: float = Field(default=1, ge=1e-9) # Kappa
    """
    The penalty factor for the noise cluster. The default value is 1. Must be greater than or equal to 1e-9.
    """

    w_prob: float = Field(default=1.0, gt=0.0) # a
    """ 
    Balancing factor, controls the influence of the probabilistic membership u in the clustering process. A higher weight increases the importance of u in updating the centroids.

    The default value is 1.0. Must be greater than 0.
    """

    def fit(self, X: NDArray) -> None:
        """
        Fits the model to the data.
        
        Parameters:
            X (NDArray): The input data.
        Returns:
            None
        """

        z = X.shape[0]
        n = X.shape[1]

        u = np.zeros((self.num_clusters, n))
        t = np.zeros((self.num_clusters, n))

        corrected_p = 1 / (self.p - 1)
        corrected_m = -2 / (self.m - 1)

        center = np.mean(X, axis=1)
        eta_sum = 0
        # print(center)
        for i in range(n):
            eta_sum += np.linalg.norm(X[:, i] - center) ** 2

        eta = (self.kappa / n) * eta_sum

        eta2 = eta
        alpha = np.full(self.num_clusters, 1.0 / self.num_clusters)

        self._create_centroids(X)

        for _ in range(self.max_iter):
            # Calculate t
            norms = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[:, :, np.newaxis], axis=0) ** 2
            exponents = (norms / (eta2 * (alpha[:, np.newaxis] ** (self.m - 1)))) ** corrected_p
            t = 1 / (1 + exponents)

            # new u
            for k in range(n):
                exact = None

                for i in range(self.num_clusters):
                    if np.linalg.norm(X[:, k] - self.centroids[:, i]) < 0.0000001:
                        exact = i
                        break

                if exact is not None:
                    u[:, k] = np.zeros(self.num_clusters)
                    u[exact, k] = 1
                    continue

                norm_diff = np.linalg.norm(X[:, k, np.newaxis] - self.centroids, axis=0)
                u[:, k] = alpha * norm_diff ** corrected_m
                u[:, k] /= np.sum(u[:, k])

            # new alpha
            alpha = np.sum((self.w_prob * u ** self.m + t ** self.p) * np.transpose(np.linalg.norm(X[:, :, np.newaxis] - self.centroids[:, np.newaxis, :], axis=0) ** 2), axis=1) ** (1 / self.m)
            alpha /= np.sum(alpha)

            # new v
            for i in range(self.num_clusters):
                sumup = np.zeros(z)
                sumdn = np.zeros(z)

                for k in range(n):
                    sumcur = (self.w_prob * u[i, k] ** self.m + t[i, k] ** self.p)
                    sumup += sumcur * X[:, k]
                    sumdn += sumcur

                self.centroids[:, i] = sumup / sumdn

        self._eta = eta2 * (alpha ** (self.m - 1))
        self._member = self.w_prob * u ** self.m + t ** self.p
        self._alpha = alpha
        self.trained = True

    @property
    def alpha(self) -> NDArray:
        if not self.is_trained():
            raise NotTrainedException()

        return self._alpha