from abc import ABC, abstractmethod


class BaseDistance(ABC):
    """Base class for distance metrics."""

    def __init__(self, metric_name="custom"):
        self.metric_name = metric_name

    @abstractmethod
    def compute(self, X, Y):
        """
        Compute the pairwise distances between two sets of vectors X and Y.

        Parameters:
        - X: ndarray of shape (n_samples_X, n_features)
        - Y: ndarray of shape (n_samples_Y, n_features)

        Returns:
        - distance_matrix: ndarray of shape (n_samples_X, n_samples_Y)
        """
        pass

    @staticmethod
    def get_supported_metrics():
        """Return the list of available built-in metrics."""
        return ["euclidean", "manhattan", "cosine"]
