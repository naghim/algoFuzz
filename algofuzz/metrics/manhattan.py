from algofuzz.metrics.base_distance import DistanceMetric
import jax.numpy as jnp


class ManhattanMetric(DistanceMetric):
    def __init__(self, metric_name="euclidean"):
        super().__init__(metric_name)

    def compute(self, X, Y):
        return jnp.linalg.norm(X[:, None, :] - Y[None, :, :], ord=1, axis=-1)
