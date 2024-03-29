from typing import List

import numpy as np


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle between two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        The first vector.
    v2 : np.ndarray
        The second vector.

    Returns
    -------
    float
        The angle between the two vectors in radians.
    """
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return angle


def turning_angles(path: np.ndarray) -> List[float]:
    """
    Calculate the turning angles between consecutive direction vectors in a path.

    Parameters
    ----------
    path : np.ndarray
        The path for which to calculate the turning angles.

    Returns
    -------
    List[float]
        A list of turning angles between consecutive direction vectors.

    Interpretation
    --------------
    Larger angles indicate more significant changes in direction, which could
    correspond to key moments in the training process where the model adjusts to
    learning new patterns or overcoming optimization challenges. Smaller angles
    suggest more gradual learning or fine-tuning phases
    """

    # Calculate direction vectors
    direction_vectors = np.diff(path, axis=0)

    # Calculate angles between consecutive direction vectors
    angles = [
        angle_between_vectors(direction_vectors[i], direction_vectors[i + 1])
        for i in range(len(direction_vectors) - 1)
    ]

    return angles
