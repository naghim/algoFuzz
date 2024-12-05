import numpy as np
import jax.numpy as jnp

"""
        Compute the pairwise distances between two sets of vectors X and Y.

        Parameters:
        - X: ndarray of shape (n_samples_X, n_features)
        - Y: ndarray of shape (n_samples_Y, n_features)

        Returns:
        - distance_matrix: ndarray of shape (n_samples_X, n_samples_Y)
"""


def euclidean(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return jnp.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)


def manhattan(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return jnp.linalg.norm(X[:, None, :] - Y[None, :, :], ord=1, axis=-1)


def cosine_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return jnp.dot(X, Y.T) / (jnp.linalg.norm(X) * jnp.linalg.norm(Y, axis=1))
