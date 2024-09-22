"""
This module contains the implementation of the Possibilistic Fuzzy C-Means Clustering algorithm proposed by Pal et al. in 2005.
"""

from numpy.typing import NDArray
from pydantic import Field
from algofuzz.fcm.base_fcm import BaseFCM
from algofuzz.fcm.eta_fcm import EtaFCM
import numpy as np

class PFCM(BaseFCM):
    """
    Partitions a numeric dataset using the Possibilistic Fuzzy C-Means Clustering (PFCM) algorithm.
    """
    preprocess_iter: int = Field(default=15, ge=1)
    """
    Number of preprocessing iterations. The default value is 15. Must be greater than or equal to 1.
    """

    p: float = Field(default=2.0, gt=1.0)
    """
    The fuzzy exponent parameter. The default value is 2.0. Must be greater than 1.
    """

    w_pos: float = Field(default=1.0, gt=0.0) # a
    """ 
    Balancing factor, controls the influence of the possibilistic membership t in the clustering process. 
    
    The default value is 1.0. Must be greater than 0.
    """

    w_prob: float = Field(default=1.0, gt=0.0) # b
    """
    Balancing factor, controls the influence of the probabilistic membership u in the clustering process. 
    
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

        self._create_centroids(X)

        eta_fcm = EtaFCM(
            num_clusters=self.num_clusters,
            m=self.m,
            centroids=self.centroids,
            max_iter=self.preprocess_iter
        )
        eta_fcm.fit(X)
        self._eta = eta_fcm.cluster_eta

        corrected_p = 1 / (self.p - 1)
        corrected_m = -2 / (self.m - 1)

        for _ in range(self.max_iter):
            # new u
            for k in range(n):
                norm_diff = np.linalg.norm(X[:, k, np.newaxis] - self.centroids, axis=0)
                exact = np.argmin(norm_diff < 0.0000001)

                exact = None

                for i in range(self.num_clusters):
                    if np.linalg.norm(X[:, k] - self.centroids[:, i]) < 0.0000001:
                        exact = i
                        break

                if exact is not None:
                    u[:, k] = np.zeros(self.num_clusters)
                    u[exact, k] = 1
                    continue

                u[:, k] = norm_diff ** corrected_m
                u[:, k] /= np.sum(u[:, k])

            # new t
            for i in range(self.num_clusters):
                for k in range(n):
                    t[i, k] = 1 / (1 + ((np.linalg.norm(X[:, k] - self.centroids[:, i]) ** 2) * self.w_prob / self._eta[i]) ** corrected_p)

            # new v
            for i in range(self.num_clusters):
                sumup = np.zeros(z)
                sumdn = np.zeros(z)

                for k in range(n):
                    sumcur = (self.w_pos * u[i, k] ** self.m + self.w_prob * t[i, k] ** self.p)
                    sumup += sumcur * X[:, k]
                    sumdn += sumcur

                self.centroids[:, i] = sumup / sumdn

        self._member = self.w_pos * u ** self.m + self.w_prob * t ** self.p
        self.trained = True