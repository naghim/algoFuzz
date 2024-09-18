import numpy as np
import math

def random_points_in_bubble(radius: float, center: np.array, num_points: int) -> np.array:
    angles = 2 * np.pi * np.random.rand(num_points)
    radii = radius * np.sqrt(np.random.rand(num_points))

    cx, cy = center
    x = cx + radii * np.cos(angles)
    y = cy + radii * np.sin(angles)

    return np.array((x, y))

def random_points_in_bubbles_grid(radii: np.array, num_points_per_circle: int, offset: int) -> np.ndarray:
    num_circles = len(radii)
    grid_size = int(math.ceil(math.sqrt(num_circles)))
    X = np.zeros((2, num_circles * num_points_per_circle))
    max_radius = np.max(radii)

    circle_counter = 0
    start = 0

    for i in range(grid_size):
        for j in range(grid_size):
            center = (i * offset * max_radius, j * offset * max_radius)  # Spacing between bubbles

            X[:, start:start + num_points_per_circle] = random_points_in_bubble(radii[circle_counter], center, num_points_per_circle)
            start += num_points_per_circle

            circle_counter += 1

            if circle_counter == num_circles:
                return X
