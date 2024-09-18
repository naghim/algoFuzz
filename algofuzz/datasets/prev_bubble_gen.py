import numpy as np

def generate_bubbles(points_in_cluster=150, offset=1.2):
    clusters = 3

    X = np.zeros((2, points_in_cluster * clusters))
    index = 0

    while index < points_in_cluster:
        r = 2 * np.random.rand(2, 1) - 1

        if r[0] * r[0] + r[1] * r[1] < 1:
            index += 1
            X[:, index-1] = r.flatten()

    while index < 2*points_in_cluster:
        r = 2 * np.random.rand(2, 1) - 1

        if r[0]*r[0] + r[1]*r[1] < 1:
            index += 1
            X[:, index-1] = (2*r + np.array([[0], [3*offset]])).flatten()

    while index < 3*points_in_cluster:
        r = 2 * np.random.rand(2, 1) - 1

        if r[0]*r[0] + r[1]*r[1] < 1:
            index += 1
            X[:, index-1] = (3*r + np.array([[4*offset], [0]])).flatten()

    return X