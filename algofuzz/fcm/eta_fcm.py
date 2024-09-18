from algofuzz.fcm.fcm import FCM
from numpy.typing import NDArray
import numpy as np

class EtaFCM(FCM):

    def fit(self, X: NDArray) -> None:
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
