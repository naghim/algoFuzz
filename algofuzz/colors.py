import random
from typing import List

__all__ = ['generate_colors']

def generate_colors(num_clusters: int, seed: int = 0) -> List[str]:
    """
    Generate a list of colors for the clusters.

    Parameters
    ----------
    num_clusters : int
        The number of clusters.
    
    seed : int
        The seed for the random number generator. The default value is 0.

    Returns
    -------
    List[str]
        A list of colors in hex format.
    """

    rand_gen = random.Random()
    rand_gen.seed(seed)

    colors = [(int(rand_gen.uniform(0, 255)), int(rand_gen.uniform(0, 255)), int(rand_gen.uniform(0, 255))) for _ in range(num_clusters)]
    hex_colors = ['#%02x%02x%02x' % color for color in colors]
    return hex_colors