from algofuzz.fcm.base_fcm import BaseFCM
from numpy.typing import NDArray
from pydantic import Field
import numpy as np
from typing import Optional

class NonoptimizedFPCM(BaseFCM):
    p: float = Field(default=2.0, gt=1.0) # Exponent
    beta: float = Field(default=1.0, gte=0.0) # Beta
    noise: Optional[float] = Field(default=0.0, gte=0.0) # temporary

    def fit(self, X: NDArray) -> None:
        if self.noise is not None:
            noisy_value = np.full((X.shape[0], 1), self.noise)
            X = np.append(X, noisy_value, axis=1)

        z = X.shape[0]
        n = X.shape[1]

        u = np.zeros((self.num_clusters, n))
        t = np.zeros((self.num_clusters, n))
        deriv_m = -1/(self.m - 1)
        deriv_p = -1/(self.p - 1)

        self._create_centroids(X)

        for _ in range(self.max_iter):
            d = np.zeros((self.num_clusters, n))

            # new u
            for k in range(n):
                for i in range(self.num_clusters):
                    for dim in range(z):
                        d[i, k] += (X[dim, k] - self.centroids[dim, i]) ** 2

            for k in range(n):
                if min(d[:, k]) < 0.0000001:
                    u[:, k] = 0

                    for i in range(self.num_clusters):
                        if d[i, k] < 0.0000001:
                            u[i, k] = 1
                else:
                    s = (d[:, k] ** deriv_m).sum()

                    for i in range(self.num_clusters):
                        u[i, k] = (d[i, k] ** deriv_m) / s

            # new t
            for i in range(self.num_clusters):
                db = np.count_nonzero(d[i, :] < 0.0000001)
                if db > 0:
                    t[i, :] = 0
                    for k in range(n):
                        if d[i, k] < 0.0000001:
                            t[i, k] = 1/db
                else:
                    s = (d[i, :] ** deriv_p).sum()

                    for k in range(n):
                        t[i, k] = (d[i, k] ** deriv_p) / s

            # new v
            for i in range(self.num_clusters):
                sumup = np.zeros(z)
                sumdn = np.zeros(z)

                for k in range(n):
                    sumcur = (u[i, k] ** self.m) + (self.beta * (t[i, k] ** self.p))
                    sumup += sumcur * X[:, k]
                    sumdn += sumcur

                self.centroids[:, i] = sumup / sumdn

        self._eta = None
        self._member = u ** self.m + self.beta * t ** self.p
        self.trained = True
