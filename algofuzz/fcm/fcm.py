"""
This module contains the implementation of the Fuzzy C-Means algorithm, proposed by Dunn in 1973 and improved by Bezdek in 1981.
"""

from algofuzz.fcm.base_fcm import BaseFCM
from numpy.typing import NDArray
from pydantic import Field
from typing import Optional
import numpy as np

class FCM(BaseFCM):
    """
    Partitions a numeric dataset using the Fuzzy C-Means (FCM) algorithm. 
    """

    kappa: float = Field(default=1.0, ge=1.0)
    """
    Regulates the severity of the penalty factor eta. The default value is 1.0. Must be greater than or equal to 1.
    """

    noise: Optional[float] = Field(default=0.0, gte=0.0) # temporary
    """ 
    (optional) A single vector will be added to the dataset. This noise vector will retain the same value given for all data points. 
    
    The default value is 0.0. Must be greater than or equal to 0.0. If None, no noise will be added.
    
    For example, if noise=2, a noise vector of [2, 2, 2, ...] will be added to the dataset based on the dimensionality of the dataset.
    """ 
    

    def fit(self, X: NDArray) -> None:
        """
        Fits the model to the data.
        
        Parameters:
            X (NDArray): The input data.
        Returns:
            None
        """

        if self.noise is not None:
            noisy_value = np.full((X.shape[0], 1), self.noise)
            X = np.append(X, noisy_value, axis=1)

        z = X.shape[0]
        n = X.shape[1]

        u = np.zeros((self.num_clusters, n))
        corrected_m = -2 / (self.m - 1)

        self._create_centroids(X)

        for _ in range(self.max_iter):
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

                s = self.calculate_initial_sum()
                for i in range(self.num_clusters):
                    u[i, k] = np.linalg.norm(X[:, k] - self.centroids[:, i]) ** corrected_m
                    s += u[i, k]

                for i in range(self.num_clusters):
                    u[i, k] /= s

            # new v
            for i in range(self.num_clusters):
                sumup = np.zeros(z)
                sumdn = np.zeros(z)

                for k in range(n):
                    sumcur = u[i, k] ** self.m
                    sumup += sumcur * X[:, k]
                    sumdn += sumcur

                self.centroids[:, i] = sumup / sumdn

        self._eta = None
        self._member = u
        self.trained = True

    def calculate_initial_sum(self):
        return 0
