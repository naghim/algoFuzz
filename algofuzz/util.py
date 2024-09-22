
import numpy as np
from algofuzz.datasets.bubble_gen import random_points_in_bubbles_grid
from algofuzz.datasets.prev_bubble_gen import generate_bubbles
from algofuzz.enums import DatasetType, FCMType
from algofuzz.fcm.base_fcm import BaseFCM
from algofuzz.fcm.fcm import FCM
from algofuzz.fcm.fcplus1m import FCPlus1M
from algofuzz.fcm.nonoptimized_fp3cm import NonoptimizedFP3CM
from algofuzz.fcm.nonoptimized_fpcm import NonoptimizedFPCM
from algofuzz.fcm.possibilistic_fcm import PFCM
from algofuzz.fcm.stpfcm import STPFCM

def load_dataset(dataset: DatasetType, num_clusters: int=3, offset: int = 1):
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
        radii = np.array(range(1, num_clusters + 1))
        num_points_per_bubble = 150
        true_labels = []

        for i in range(0, num_clusters):
            true_labels.extend([i] * num_points_per_bubble)

        return random_points_in_bubbles_grid(radii, num_points_per_bubble, offset), num_clusters, true_labels
    elif dataset == DatasetType.PrevBubbles:
        print('Prev BUBBLES?')
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

def get_fcm_by_type(fcm_type: FCMType | str) -> BaseFCM:
    if isinstance(fcm_type, str):
        fcm_type = FCMType[fcm_type]

    if fcm_type == FCMType.FCM:
        return FCM
    elif fcm_type == FCMType.FCPlus1M:
        return FCPlus1M
    elif fcm_type == FCMType.STPFCM:
        return STPFCM
    elif fcm_type == FCMType.PFCM:
        return PFCM
    elif fcm_type == FCMType.NonoptimizedFP3CM:
        return NonoptimizedFP3CM
    elif fcm_type == FCMType.NonoptimizedFPCM:
        return NonoptimizedFPCM
