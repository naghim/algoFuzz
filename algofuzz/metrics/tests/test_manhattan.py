import pytest
import numpy as np
import os
from algofuzz.metrics import ManhattanDistance

group = "ManhattanDistance"


@pytest.fixture
def vectors():
    np.random.seed(0)
    n = 10**2
    m = 15000
    vector_a = np.random.rand(n, m)
    vector_b = np.random.rand(n, m)
    return vector_a, vector_b


def test_2d_distance():
    point1 = np.array([[0, 0]])
    point2 = np.array([[3, 4]])

    m = ManhattanDistance()
    print(m.compute(point1, point2))
    np.testing.assert_allclose(
        m.compute(point1, point2), np.array([[7]]), atol=1e-9
    )


def test_3d_distance():
    point1 = np.array([[1, 2, 3]])
    point2 = np.array([[4, 5, 6]])

    m = ManhattanDistance()
    np.testing.assert_allclose(
        m.compute(point1, point2), np.array([[9]]), atol=1e-9
    )


def test_same_point():
    point1 = np.array([[1, 1, 1]])
    point2 = np.array([[1, 1, 1]])

    m = ManhattanDistance()
    np.testing.assert_allclose(
        m.compute(point1, point2), np.array([[0]]), atol=1e-9
    )


def test_large_numbers():
    point1 = np.array([[1e6, 1e6]])
    point2 = np.array([[2e6, 2e6]])

    m = ManhattanDistance()
    np.testing.assert_allclose(
        m.compute(point1, point2), np.array([[2e6]]), atol=1e-9
    )


def test_n_dimensional_distance():
    point1 = np.array([[1, 2, 3, 4, 5]])
    point2 = np.array([[5, 4, 3, 2, 1]])

    m = ManhattanDistance()
    np.testing.assert_allclose(
        m.compute(point1, point2), np.array([[12]]), atol=1e-9
    )


def test_negative_values():
    point1 = np.array([[-1, -2, -3]])
    point2 = np.array([[1, 2, 3]])
    m = ManhattanDistance()

    np.testing.assert_allclose(
        m.compute(point1, point2), np.array([[12]]), atol=1e-9
    )


@pytest.mark.benchmark(group=group)
@pytest.mark.skipif(os.environ.get("BENCHMARK") != "1", reason="Skipping benchmarks")
def test_benchmark_distance_np(benchmark, vectors):

    vector_a, vector_b = vectors

    def numpy_function(X, Y):
        return np.linalg.norm(
            X[:, np.newaxis, :] - Y[np.newaxis, :, :], ord=0, axis=-1)
    benchmark(numpy_function, vector_a, vector_b)


@pytest.mark.benchmark(group=group)
@pytest.mark.skipif(os.environ.get("BENCHMARK") != "1", reason="Skipping benchmarks")
def test_benchmark_distance_current_implementation(benchmark, vectors):

    vector_a, vector_b = vectors
    m = ManhattanDistance()
    benchmark(m.compute, vector_a, vector_b)
