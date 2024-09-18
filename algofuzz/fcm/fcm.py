from algofuzz.fcm.base_fcm import BaseFCM
from numpy.typing import NDArray
from pydantic import Field
from typing import Optional
import numpy as np

class FCM(BaseFCM):
    kappa: float = Field(default=1.0, gt=1.0) # Kappa
    noise: Optional[float] = Field(default=0.0, gte=0.0) # temporary

    def fit(self, X: NDArray) -> None:
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
