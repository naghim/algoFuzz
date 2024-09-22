"""
This model is an extension of the FCM model that includes a penalty term (eta) for each cluster: provides the default implementation of the fit method that calculates the eta parameter for each cluster based on the membership matrix and the centroids of the clusters. 
"""

from algofuzz.fcm.fcm import FCM
from numpy.typing import NDArray
import numpy as np

class EtaFCM(FCM):
    """
    An extension of the FCM model that includes a penalty term (eta) for each cluster depending on the distance between the data points and the centroids of the clusters.
    """
    def fit(self, X: NDArray) -> None:
        """
        Fits the model to the data.
        
        Parameters:
            X (NDArray): The input data.
        Returns:
            None
        """
        FCM.fit(self, X)

        n = X.shape[1]
        self._eta = np.zeros(self.num_clusters)

        for i in range(self.num_clusters):
            # eta
            up = 0
            dn = 0

            for k in range(n):
                up += self.member[i, k] ** self.m * (np.linalg.norm(X[:, k] - self.centroids[:, i])) ** 2
                dn += self.member[i, k] ** self.m

            self._eta[i] = (up / dn) * self.kappa
