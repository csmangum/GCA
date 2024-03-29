import numpy as np

from path_complexity.total_path_length import total_path_length


def detour_factor(path: np.ndarray) -> float:
    """
    Calculate the detour factor of a trajectory.

    The detour factor is a measure of how much longer the path is compared to a
    straight-line distance between the start and end points of the trajectory.
    A higher detour factor indicates a more circuitous path.

    Parameters
    ----------
    path : np.ndarray
        The path to calculate the detour factor of.

    Returns
    -------
    float
        The detour factor of the trajectory.

    Interpretation
    --------------
    A detour factor slightly greater than 1 suggests minimal detours, indicating
    a relatively direct path of optimization.
    """

    # Calculate the straight-line distance between the start and end points of the trajectory
    straight_line_distance = np.linalg.norm(path[-1] - path[0])

    # Calculate the total path length of the trajectory
    total_path_length = total_path_length(path)

    # Calculate the detour factor
    detour_factor = total_path_length / straight_line_distance

    return detour_factor
