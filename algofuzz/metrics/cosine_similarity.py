from algofuzz.metrics import BaseDistance
import jax.numpy as jnp


class CosineSimilarity(BaseDistance):
    def __init__(self, metric_name="cosine_similarity"):
        super().__init__(metric_name)

    def compute(self, X, Y):
        return jnp.dot(X, Y.T) / (jnp.linalg.norm(X) * jnp.linalg.norm(Y, axis=1))
