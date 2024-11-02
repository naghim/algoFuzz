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

    def fit_new(self, X: NDArray) -> None:
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
        # print(center)

        eta_sum = np.linalg.norm(X[:, :] - center) ** 2  # np.sum()
        # print(eta_sum)

        eta = (self.kappa / n) * eta_sum

        eta2 = eta
        alpha = np.full(self.num_clusters, 1.0 / self.num_clusters)
        # print(alpha)

        self._create_centroids(X, transpose=False)

        # checked center,eta_sum,alpha
        for _ in range(self.max_iter):
            distances = np.linalg.norm(
                X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
            # print(norms)
            exponents = (
                (distances ** 2) / (eta2 * (alpha[np.newaxis, :] ** (self.m - 1)))) ** corrected_p
            t = 1 / (1 + exponents)
            # print(distances**2, exponents, t)
            # checked distances,exponents,t
            # print(distances.shape)
            exact_matches = np.isclose(distances, 0, atol=1e-7)
            u = np.zeros((n, self.num_clusters))

            if np.any(exact_matches):
                exact_indices = np.argmax(exact_matches, axis=1)
                u[np.arange(n), exact_indices] = 1
            non_exact_indices = np.where(~np.any(exact_matches, axis=1))[0]
            # print(distances)
            # print(u.shape, distances.shape)
            if non_exact_indices.size > 0:
                # shape: (num_non_exact_samples, num_clusters)
                norm_diff_non_exact = distances[non_exact_indices, :]
                # shape: (, num_non_exact_samples)
                u[non_exact_indices, :] = alpha * \
                    (norm_diff_non_exact ** corrected_m)
                u[non_exact_indices, :] /= np.sum(
                    u[non_exact_indices, :], axis=1, keepdims=True)
                # print(u[non_exact_indices, :])

            # print(u)
            # return
            # Transposed
            # alpha = np.sum((self.w_prob * u ** self.m + t ** self.p) * np.transpose(np.linalg.norm(
            # X[:, :, np.newaxis] - self.centroids[:, np.newaxis, :], axis=0) ** 2), axis=1) ** (1 / self.m)
            # alpha /= np.sum(alpha)
            # Original
            alpha = np.sum((self.w_prob * u ** self.m + t ** self.p)
                           * (distances ** 2), axis=0) ** (1 / self.m)
            # print(alpha)
            alpha /= np.sum(alpha)

            # alpha checked
            # print(alpha)
            # return
            # Transposed
            #  for i in range(self.num_clusters):
            # sumup = np.zeros(z)
            # sumdn = np.zeros(z)
#
            # for k in range(n):
            # sumcur = (self.w_prob * u[i, k] **
            #   self.m + t[i, k] ** self.p)
            # sumup += sumcur * X[:, k]
            # sumdn += sumcur
#
            # self.centroids[:, i] = sumup / sumdn
            # ORiginal

            sumcur = (self.w_prob * u **
                      self.m + t ** self.p)
            X_transposed = X.T
            results = sumcur.T @ X.T
            example_array = X[0, :]
            # sumdn = np.sum(sumcur, axis=0)
            for k in range(self.num_clusters):
                sumup = np.zeros(z)
                sumdn = np.zeros(z)
                for i in range(n):
                    sumup += sumcur[i, k] * X[i, :]
                    sumdn += sumcur[i, k]
                    # Something aint right here
                self.centroids[k, :] = sumup / sumdn
            # print(u.shape, t.shape)
            # print(self.w_prob, self.p, self.m)
            # return
            # print(sumcur)
            # Sumup shape (5,4)
            # sum_X = np.sum(X, axis=0)
            # sumdn = np.sum(sumcur, axis=0)
            # sumup = sumdn[:, np.newaxis] * sum_X[np.newaxis, :] / 10
            # print(sumup)
            # print(sumdn)
            # print(sum_X)
            # return
            # self.centroids = sumup / sumdn[:, np.newaxis]
            # print(self.centroids)

        self._eta = eta2 * (alpha ** (self.m - 1))
        self._member = self.w_prob * u ** self.m + t ** self.p
        self._alpha = alpha
        self.trained = True

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
        # print(center)

        eta_sum = 0
        # print(center)
        for i in range(n):
            eta_sum += np.linalg.norm(X[:, i] - center) ** 2

        # print(eta_sum)
        # return

        eta = (self.kappa / n) * eta_sum

        eta2 = eta
        alpha = np.full(self.num_clusters, 1.0 / self.num_clusters)
        # print(alpha)

        self._create_centroids(X)

        for _ in range(self.max_iter):
            # Calculate t
            norms = np.linalg.norm(
                X[:, np.newaxis, :] - self.centroids[:, :, np.newaxis], axis=0) ** 2
            # print(norms)
            exponents = (
                norms / (eta2 * (alpha[:, np.newaxis] ** (self.m - 1)))) ** corrected_p
            t = 1 / (1 + exponents)
            # print(norms, exponents, t)
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

                norm_diff = np.linalg.norm(
                    X[:, k, np.newaxis] - self.centroids, axis=0)
                # print(norms, norm_diff**2)

                u[:, k] = alpha * norm_diff ** corrected_m
                u[:, k] /= np.sum(u[:, k])
                # print(u[:, k])

            # print(u)
            # print(u.T)
            # return
            # U checked
            # return
            # new alpha
            # print(u.shape, t.shape)
            alpha = np.sum((self.w_prob * u ** self.m + t ** self.p) * np.transpose(np.linalg.norm(
                X[:, :, np.newaxis] - self.centroids[:, np.newaxis, :], axis=0) ** 2), axis=1) ** (1 / self.m)
            # print(alpha)
            alpha /= np.sum(alpha)
            # print(alpha)
            # return
            # new v
            sumcur_m = np.zeros((self.num_clusters, n))
            sumup_m = np.zeros((self.num_clusters, z))
            sumdn_m = np.zeros((self.num_clusters))
            print(self.w_prob, self.p, self.m)
            for i in range(self.num_clusters):
                sumup = np.zeros(z)
                sumdn = 0  # np.zeros(z)

                for k in range(n):
                    sumcur = ((self.w_prob * (u[i, k] **
                              self.m)) + (t[i, k] ** self.p))
                    # sumcur = (self.w_prob * u[k,i] **
                    #   self.m + t[k,i] ** self.p)
                    sumup += sumcur * X[:, k]
                    print(X[:, k])
                    sumdn += sumcur
                    sumcur_m[i, k] = sumcur
                sumup_m[i] = sumup
                # self.centroids[:, i] = sumup / sumdn
                sumdn_m[i] = sumdn
            print('----')
            # print(sumcur_m)
            # print(sumup_m)
            # print(sumdn_m)
            return
            # print(self.centroids)
        return
        self._eta = eta2 * (alpha ** (self.m - 1))
        self._member = self.w_prob * u ** self.m + t ** self.p
        self._alpha = alpha
        self.trained = True


if __name__ == '__main__':
    from algofuzz.enums import DatasetType, CentroidStrategy
    from algofuzz.datasets import load_dataset
    np.random.seed(0)
    st_1 = STPFCM()
#
    # x, c, true_labels = load_dataset(DatasetType.Iris, tranpose=True)
    # print(x, x.shape)
    # st_1.fit(x[:, :10])
    # centroids = st_1.centroids
    # st_2 = STPFCM(centroids=centroids.T)
    x, c, true_labels = load_dataset(DatasetType.Iris, tranpose=False)
    # print(x, x.shape)
#
    st_1.fit_new(x)
    # st_1.evaluate(true_labels)
