import pytest
import numpy as np
import os
from algofuzz.metrics.distance_pairwise import manhattan

group = "ManhattanDistance"


def numpy_function(X, Y):
    return np.abs(X[:, None] - Y).sum(-1)


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

    np.testing.assert_allclose(
        manhattan(point1, point2), np.array([[7]]), atol=1e-9
    )


def test_3d_distance():
    point1 = np.array([[1, 2, 3]])
    point2 = np.array([[4, 5, 6]])

    np.testing.assert_allclose(
        manhattan(point1, point2), np.array([[9]]), atol=1e-9
    )


def test_same_point():
    point1 = np.array([[1, 1, 1]])
    point2 = np.array([[1, 1, 1]])

    np.testing.assert_allclose(
        manhattan(point1, point2), np.array([[0]]), atol=1e-9
    )


def test_large_numbers():
    point1 = np.array([[1e6, 1e6]])
    point2 = np.array([[2e6, 2e6]])

    np.testing.assert_allclose(
        manhattan(point1, point2), np.array([[2e6]]), atol=1e-9
    )


def test_n_dimensional_distance():
    point1 = np.array([[1, 2, 3, 4, 5]])
    point2 = np.array([[5, 4, 3, 2, 1]])

    np.testing.assert_allclose(
        manhattan(point1, point2), np.array([[12]]), atol=1e-9
    )


def test_negative_values():
    point1 = np.array([[-1, -2, -3]])
    point2 = np.array([[1, 2, 3]])

    np.testing.assert_allclose(
        manhattan(point1, point2), np.array([[12]]), atol=1e-9
    )


def test_pairwise_distance():
    points1 = np.array([[-1, -2, -3], [1, 1, 1], [3, 3, 3]])
    points2 = np.array([[1, 2, 3], [-1, -1, -1], [4, 5, 6]])

    pairwise_distances = manhattan(points1, points2)
    expected_distances = numpy_function(points1, points2)

    np.testing.assert_allclose(
        pairwise_distances, expected_distances, atol=1e-9)


@pytest.mark.benchmark(group=group)
@pytest.mark.skipif(os.environ.get("BENCHMARK") != "1", reason="Skipping benchmarks")
def test_benchmark_distance_np(benchmark, vectors):

    vector_a, vector_b = vectors

    benchmark(numpy_function, vector_a, vector_b)


@pytest.mark.benchmark(group=group)
@pytest.mark.skipif(os.environ.get("BENCHMARK") != "1", reason="Skipping benchmarks")
def test_benchmark_distance_current_implementation(benchmark, vectors):

    vector_a, vector_b = vectors

    benchmark(manhattan, vector_a, vector_b)
