"""
This module contains datasets that can be used to evaluate clustering algorithms.

The datasets are categorized into two types:

1. Toy Datasets:
    - ``bubble_gen``: Generates artificial testing scenarios for clustering algorithms.
    - ``prev_bubble_gen``: Another simpler generator for creating artificial testing scenarios.

2. Real-life Datasets:
    - ``glass``: A dataset containing the refractive index and the amounts of eight elements in glass.

These datasets are designed to help in testing and evaluating the performance of various clustering algorithms.
"""
import numpy as np
from algofuzz.datasets.bubble_gen import random_points_in_bubbles_grid
from algofuzz.datasets.prev_bubble_gen import generate_bubbles
from algofuzz.enums import DatasetType

__all__ = ['load_dataset']

def load_dataset(dataset: DatasetType, num_clusters: int = 3, offset: int = 1):
    """
    Load a dataset based on the specified type.

    Parameters:
        dataset (DatasetType): The type of dataset to load.
        num_clusters (int): The number of clusters in the dataset. The default value is 3.
        offset (int): The spacing between bubbles in unit dimensions. The default value is 1.
    
    Returns:
        data: np.ndarray
            The dataset.
        num_clusters: int
            The number of clusters in the dataset.
        true_labels: np.ndarray
            The true labels of the dataset
    """
    if dataset == DatasetType.Iris:
        from sklearn.datasets import load_iris
        iris = load_iris()
        data = iris.data
        true_labels = iris.target
        num_clusters = len(iris.target_names)
        return data.T, num_clusters, true_labels
    if dataset == DatasetType.Glass:
        from algofuzz.datasets.glass import load_glass
        glass = load_glass()
        data = glass.data
        true_labels = glass.target
        num_clusters = len(glass.target_names)
        return data.T, num_clusters, true_labels
    if dataset == DatasetType.Seeds:
        from algofuzz.datasets.seeds import load_seeds
        seeds = load_seeds()
        data = seeds.data
        true_labels = seeds.target
        num_clusters = len(seeds.target_names)
        return data.T, num_clusters, true_labels
    elif dataset == DatasetType.NormalizedIris:
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import MinMaxScaler
        iris = load_iris()
        scaler = MinMaxScaler()
        data = scaler.fit_transform(iris.data)
        true_labels = iris.target
        num_clusters = len(iris.target_names)
        return data.T, num_clusters, true_labels
    if dataset == DatasetType.NormalizedGlass:
        from algofuzz.datasets.glass import load_glass
        from sklearn.preprocessing import MinMaxScaler
        glass = load_glass()
        scaler = MinMaxScaler()
        data = scaler.fit_transform(glass.data)
        true_labels = glass.target
        num_clusters = len(glass.target_names)
        return data.T, num_clusters, true_labels
    if dataset == DatasetType.NormalizedSeeds:
        from algofuzz.datasets.seeds import load_seeds
        from sklearn.preprocessing import MinMaxScaler
        seeds = load_seeds()
        scaler = MinMaxScaler()
        data = scaler.fit_transform(seeds.data)
        true_labels = seeds.target
        num_clusters = len(seeds.target_names)
        return data.T, num_clusters, true_labels
    elif dataset == DatasetType.NoisyNormalizedIris:
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import MinMaxScaler
        iris = load_iris()
        scaler = MinMaxScaler()
        data = scaler.fit_transform(iris.data)
        true_labels = iris.target
        num_clusters = len(iris.target_names)

        noisy_value = 30
        noisy_value = np.full(data.shape[1], noisy_value)
        data = np.vstack((data, noisy_value))

        true_labels = np.append(true_labels, max(iris.target))

        return data.T, num_clusters, true_labels
    elif dataset == DatasetType.Bubbles:
        num_clusters = 3
        radii = np.array(range(1, num_clusters + 1))
        num_points_per_bubble = 150
        true_labels = []

        for i in range(0, num_clusters):
            true_labels.extend([i] * num_points_per_bubble)

        return random_points_in_bubbles_grid(radii, num_points_per_bubble, offset), num_clusters, true_labels
    elif dataset == DatasetType.PrevBubbles:
        num_clusters = 3
        num_points_per_bubble = 150
        true_labels = []

        for i in range(0, num_clusters):
            true_labels.extend([i] * num_points_per_bubble)

        return generate_bubbles(num_points_per_bubble, offset), num_clusters, true_labels
    elif dataset == DatasetType.Wine:
        from sklearn.datasets import load_wine
        wine = load_wine()
        data = wine.data
        true_labels = wine.target
        return wine.data.T, num_clusters, true_labels
    elif dataset == DatasetType.NormalizedWine:
        from sklearn.datasets import load_wine
        from sklearn.preprocessing import MinMaxScaler
        wine = load_wine()
        scaler = MinMaxScaler()
        data = scaler.fit_transform(wine.data)
        true_labels = wine.target
        return data.T, num_clusters, true_labels
    elif dataset == DatasetType.BreastCancer:
        from sklearn.datasets import load_breast_cancer
        bcancer = load_breast_cancer()
        data = bcancer.data
        true_labels = bcancer.target
        num_clusters = len(bcancer.target_names)
        return data.T, num_clusters, true_labels
    elif dataset == DatasetType.NormalizedBreastCancer:
        from sklearn.datasets import load_breast_cancer
        from sklearn.preprocessing import MinMaxScaler
        bcancer = load_breast_cancer()
        scaler = MinMaxScaler()
        data = scaler.fit_transform(bcancer.data)
        true_labels = bcancer.target
        num_clusters = len(bcancer.target_names)
        return data.T, num_clusters, true_labels
    elif dataset == DatasetType.Bubbles1:
        return load_dataset(DatasetType.PrevBubbles, num_clusters=3, offset=1)
    elif dataset == DatasetType.Bubbles2:
        return load_dataset(DatasetType.PrevBubbles, num_clusters=3, offset=1.05)
    elif dataset == DatasetType.Bubbles3:
        return load_dataset(DatasetType.PrevBubbles, num_clusters=3, offset=1.1)
    elif dataset == DatasetType.Bubbles4:
        return load_dataset(DatasetType.PrevBubbles, num_clusters=3, offset=1.2)
    else:
        raise ValueError('Invalid dataset type')
