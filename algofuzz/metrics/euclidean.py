from algofuzz.metrics import BaseDistance
import jax.numpy as jnp


class EuclideanDistance(BaseDistance):
    def __init__(self, metric_name="euclidean"):
        super().__init__(metric_name)

    def compute(self, X, Y):
        return jnp.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)
