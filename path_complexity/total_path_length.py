import numpy as np


def total_path_length(path: np.ndarray) -> float:
    """
    Calculate the total path length of a path of vectors.

    Parameters
    ----------
    path : np.ndarray
        The path to calculate the total path length of.

    Returns
    -------
    float
        The total path length of the path.
    """
    distances = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
    total_path_length = np.sum(distances)

    return total_path_length
