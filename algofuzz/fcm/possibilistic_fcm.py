from numpy.typing import NDArray
from pydantic import Field
from algofuzz.fcm.base_fcm import BaseFCM
from algofuzz.fcm.eta_fcm import EtaFCM
import numpy as np

class PFCM(BaseFCM):
    preprocess_iter: int = Field(default=15, ge=1)
    p: float = Field(default=2.0, gt=1.0) # Exponent
    weight: float = Field(default=1.0, gt=0.0) # a
    b: float = Field(default=1.0, gt=0.0) # b

    def fit(self, X: NDArray) -> None:
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
                    t[i, k] = 1 / (1 + ((np.linalg.norm(X[:, k] - self.centroids[:, i]) ** 2) * self.b / self._eta[i]) ** corrected_p)

            # new v
            for i in range(self.num_clusters):
                sumup = np.zeros(z)
                sumdn = np.zeros(z)

                for k in range(n):
                    sumcur = (self.weight * u[i, k] ** self.m + self.b * t[i, k] ** self.p)
                    sumup += sumcur * X[:, k]
                    sumdn += sumcur

                self.centroids[:, i] = sumup / sumdn

        self._member = self.weight * u ** self.m + self.b * t ** self.p
        self.trained = True