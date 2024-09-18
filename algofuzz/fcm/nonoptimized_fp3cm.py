from numpy.typing import NDArray
from pydantic import Field
from algofuzz.fcm.base_fcm import BaseFCM
from algofuzz.exceptions import NotTrainedException
import numpy as np

class NonoptimizedFP3CM(BaseFCM):
    p: float = Field(default=2.0, gt=1.0) # Exponent
    eta: float = Field(default=0.1, ge=1e-9) # Learning rate

    def fit(self, X: NDArray) -> None:
        # x, y dimensions
        z = X.shape[0]
        n = X.shape[1]

        u = np.zeros((self.num_clusters, n))
        t = np.zeros((self.num_clusters, n))

        self._create_centroids(X)

        corrected_p = 1 / (self.p - 1)
        corrected_m = -1 / (self.m - 1)

        eta2 = self.eta ** 2

        for _ in range(self.max_iter):
            # new t
            for i in range(self.num_clusters):
                for k in range(n):
                    t[i, k] = 1 / (1 + ((np.linalg.norm(X[:, k] - self.centroids[:, i]) ** 2) / eta2) ** corrected_p)

            # new u
            for k in range(n):
                szum = 0
                exact = None

                for i in range(self.num_clusters):
                    if np.linalg.norm(X[:, k] - self.centroids[:, i]) < 0.0000001:
                        exact = i
                        break

                if exact is not None:
                    u[:, k] = np.zeros(self.num_clusters)
                    u[exact, k] = 1
                    continue

                for i in range(self.num_clusters):
                    u[i, k] = ((t[i, k] ** self.p) * (np.linalg.norm(X[:, k] - self.centroids[:, i]) ** 2) + eta2 * (1-t[i, k]) ** self.p) ** corrected_m
                    szum += u[i, k]

                for i in range(self.num_clusters):
                    u[i, k] /= szum

            # new v
            for i in range(self.num_clusters):
                sumup = np.zeros(z)
                sumdn = np.zeros(z)

                for k in range(n):
                    sumcur = (u[i, k] ** self.m) * (t[i, k] ** self.p)
                    sumup += sumcur * X[:, k]
                    sumdn += sumcur

                self.centroids[:, i] = sumup / sumdn

        self._member = u ** self.m + t ** self.p
        self.trained = True