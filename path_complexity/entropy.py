import numpy as np
from scipy.stats import entropy


def entropy(path: np.ndarray) -> float:
    """
    Calculate the entropy of the directions in a trajectory.

    The entropy of the directions in a trajectory is a measure of the randomness
    or predictability of the changes in direction between consecutive vectors.

    Parameters
    ----------
    path : np.ndarray
        The trajectory to calculate the entropy of.

    Returns
    -------
    float
        The entropy of the directions in the trajectory.

    Interpretation
    --------------
    Higher entropy values would indicate a greater diversity in direction changes,
    while lower values would suggest more uniformity.
    """
    # Calculate the differences (direction changes) between consecutive vectors
    directions = np.diff(path, axis=0)

    # Normalize the direction vectors to compute unique directions accurately
    unit_directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]

    # Convert unit directions to a hashable type to count unique directions
    direction_tuples = [tuple(dir) for dir in unit_directions]

    # Count occurrences of each unique direction
    _, counts = np.unique(direction_tuples, axis=0, return_counts=True)

    # Calculate the probability of each direction
    probabilities = counts / counts.sum()

    # Calculate and return the entropy
    return entropy(probabilities)
