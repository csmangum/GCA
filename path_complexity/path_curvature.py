from typing import Tuple

import numpy as np


def calculate_angles(path: np.ndarray) -> np.ndarray:
    """
    Calculate the angles (in radians) between consecutive vectors in a path.

    Parameters
    ----------
    path : np.ndarray
        The path to calculate the angles between consecutive vectors of.

    Returns
    -------
    np.ndarray
        An array containing the angles (in radians) between consecutive vectors.
    """
    # Calculate vector differences (directions) between consecutive vectors
    directions = np.diff(path, axis=0)

    # Normalize the direction vectors to unit vectors
    unit_directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]

    # Calculate the dot product between consecutive unit direction vectors
    dot_products = np.einsum("ij,ij->i", unit_directions[:-1], unit_directions[1:])

    # Ensure the dot product values are within the valid range for arccos ([-1, 1])
    dot_products = np.clip(dot_products, -1.0, 1.0)

    # Calculate angles in radians between consecutive directions
    angles = np.arccos(dot_products)

    return angles


def path_curvature(path: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the curvature of a path of vectors.

    The curvature of the path can provide insights into how quickly the path
    changes direction. High curvature indicates more abrupt changes in the
    direction of weight updates, suggesting a more complex optimization landscape
    or learning dynamics.

    Parameters
    ----------
    path : np.ndarray
        The path to calculate the curvature of.

    Returns
    -------
    Tuple[float, float]
        A tuple containing the mean and maximum curvature along the path.

    Interpretation
    --------------
    A lower mean angle suggests a relatively straight path, and a higher maximum
    angle indicates a significant change in direction at least at one point along
    the trajectory.
    """
    # Calculate the angles (in radians) representing the curvature at each point along the path
    angles = calculate_angles(path)

    # Calculate the mean and maximum curvature along the path
    mean_angle = np.mean(angles)
    max_angle = np.max(angles)

    return mean_angle, max_angle
