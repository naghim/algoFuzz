# test_clustering.py
import pytest
import numpy as np
from algofuzz.fcm import NonoptimizedSTPFCM, STPFCM
from algofuzz.enums import DatasetType, CentroidStrategy
from algofuzz.datasets import load_dataset


@pytest.fixture
def dataset():

    X, c, true_labels = load_dataset(DatasetType.Iris)
    return X, c, true_labels


@pytest.fixture
def dataset_notT():
    x, c, true_labels = load_dataset(DatasetType.Iris, tranpose=False)
    return x, c, true_labels


@pytest.fixture
def parameters():
    np.random.seed(0)
    m = 2  # Fuzzifier parameter
    p = 2  # Exponent parameter
    steps = 150  # Number of iterations
    centroid_strategy = CentroidStrategy.Random
    return m, p, steps, centroid_strategy


def test_NonOptimizedSTPFCM(benchmark, dataset, parameters):
    np.random.seed(0)
    data, clus_num, true_labels = dataset
    m, p, steps, centroid_strategy = parameters
    expected_evaluate_result = np.asarray(
        (0.92, 0.7859265306122449, 0.7773403198092464), dtype=np.float64)
    expected_labels = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                  2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    fcm = NonoptimizedSTPFCM(num_clusters=clus_num,
                             m=m,
                             p=p,
                             max_iter=steps,
                             centroid_strategy=centroid_strategy)

    benchmark(fcm.fit, data)
    evaluate_result = np.asarray(fcm.evaluate(true_labels))
    print(evaluate_result)
    print(expected_evaluate_result)
    assert fcm is not None
    assert fcm.trained is True
    assert len(fcm.labels) == data.shape[1]
    assert np.array_equal(fcm.labels, expected_labels)
    np.testing.assert_allclose(
        evaluate_result, expected_evaluate_result)


def test_STPFCM(benchmark, dataset_notT, parameters):
    np.random.seed(0)

    data, clus_num, true_labels = dataset_notT
    m, p, steps, centroid_strategy = parameters

    expected_evaluate_result = np.asarray(
        (0.92, 0.7859265306122449, 0.7773403198092464), dtype=np.float64)
    expected_labels = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                  2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    fcm = STPFCM(num_clusters=clus_num,
                 m=m,
                 p=p,
                 max_iter=steps,
                 centroid_strategy=centroid_strategy, transposed=False)

    benchmark(fcm.fit, data)
    evaluate_result = np.asarray(fcm.evaluate(true_labels))
    print(evaluate_result)
    print(expected_evaluate_result)
    assert fcm is not None
    assert fcm.trained is True
    assert len(fcm.labels) == data.shape[0]
    assert np.array_equal(fcm.labels, expected_labels)
    np.testing.assert_allclose(
        evaluate_result, expected_evaluate_result)
