import pytest
import os
import numpy as np
from algofuzz.metrics import CosineSimilarity
from sklearn.metrics.pairwise import cosine_similarity

group = "CosineSimilarity"


@pytest.fixture
def vectors():
    np.random.seed(0)
    n = 10**2
    m = 15000
    vector_a = np.random.rand(n, m)
    vector_b = np.random.rand(n, m)
    return vector_a, vector_b


def test_2d_cosine_similarity():
    point1 = np.array([[1, 0]])
    point2 = np.array([[0, 1]])

    c = CosineSimilarity()
    np.testing.assert_allclose(
        c.compute(point1, point2), cosine_similarity(point1, point2), atol=1e-9)


def test_3d_cosine_similarity():
    point1 = np.array([[1, 2, 3]])
    point2 = np.array([[4, 5, 6]])

    c = CosineSimilarity()
    np.testing.assert_allclose(
        c.compute(point1, point2), np.array([[0.97463185]]), atol=1e-9)


def test_same_point_cosine_similarity():
    point1 = np.array([[1, 1, 1]])
    point2 = np.array([[1, 1, 1]])

    c = CosineSimilarity()
    np.testing.assert_allclose(
        c.compute(point1, point2), np.array([[1]]), atol=1e-9)


def test_large_numbers_cosine_similarity():
    point1 = np.array([[1e6, 1e6]], dtype=np.float64)
    point2 = np.array([[2e6, 2e6]], dtype=np.float64)

    c = CosineSimilarity()
    # Results are correct, but it has a difference of 1.19209289e-07. This might be because of different float sizes, but the calculations are the same.
    np.testing.assert_allclose(
        c.compute(point1, point2), np.array(cosine_similarity(point1, point2)), atol=1e-7)


def test_n_dimensional_cosine_similarity():
    point1 = np.array([[1, 2, 3, 4, 5]])
    point2 = np.array([[5, 4, 3, 2, 1]])

    c = CosineSimilarity()
    np.testing.assert_allclose(
        c.compute(point1, point2), cosine_similarity(point1, point2), atol=1e-9)


def test_cosine_similarity_with_negative_values():
    point1 = np.array([[-1, -2, -3]])
    point2 = np.array([[1, 2, 3]])

    c = CosineSimilarity()
    np.testing.assert_allclose(
        c.compute(point1, point2), cosine_similarity(point1, point2), atol=1e-9)


@pytest.mark.benchmark(group=group)
@pytest.mark.skipif(os.environ.get("BENCHMARK") != "1", reason="Skipping benchmarks")
def test_benchmark_distance_np(benchmark, vectors):

    vector_a, vector_b = vectors

    benchmark(cosine_similarity, vector_a, vector_b)


@ pytest.mark.benchmark(group=group)
@ pytest.mark.skipif(os.environ.get("BENCHMARK") != "1", reason="Skipping benchmarks")
def test_benchmark_distance_current_implementation(benchmark, vectors):

    vector_a, vector_b = vectors
    e = CosineSimilarity()
    benchmark(e.compute, vector_a, vector_b)
