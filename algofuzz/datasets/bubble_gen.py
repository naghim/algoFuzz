"""
This module contains functions to generate random datasets comprised of multiple clusters in the form of bubbles.

It can be used to generate datasets for clustering algorithms, where the points are
clustered around the center of each bubble. The bubbles can have different radii and
they can also be placed on a grid.
"""
import numpy as np
import math

def random_points_in_bubble(radius: float, center: np.array, num_points: int) -> np.array:
    """
    Generate random two-dimensional points in a bubble.

    Parameters:
        radius (float): The radius of the bubble.
        center (np.array): The center of the bubble.
        num_points (int): The number of random points to generate.
    
    Returns
    -------
    data: np.ndarray
        An array of shape ``(2, num_points)`` containing the random points.
    """
    angles = 2 * np.pi * np.random.rand(num_points)
    radii = radius * np.sqrt(np.random.rand(num_points))

    cx, cy = center
    x = cx + radii * np.cos(angles)
    y = cy + radii * np.sin(angles)

    return np.array((x, y))

def random_points_in_bubbles_grid(radii: np.array, num_points_per_circle: int, offset: int) -> np.ndarray:
    """
    Generate random two-dimensional points in multiple bubbles placed on a grid.

    Parameters:
        radii (np.array): An array of bubble radii. There will be one bubble for each radius.
        num_points_per_circle (int): The number of random points to generate in each bubble.
        offset (int): The spacing between bubbles in unit dimensions.
    
    Returns
    -------
    data: np.ndarray
        An array of shape ``(2, num_points_per_circle * num_circles)`` containing the random points.
    """
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
