from algofuzz.metrics.base_distance import BaseDistance
import jax.numpy as jnp


class ManhattanDistance(BaseDistance):
    def __init__(self, metric_name="manhattan"):
        super().__init__(metric_name)

    def compute(self, X, Y):
        return jnp.linalg.norm(X[:, None, :] - Y[None, :, :], ord=1, axis=-1)
