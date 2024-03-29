from typing import List

import numpy as np
from sklearn.linear_model import LinearRegression


def box_count(path: np.ndarray, sizes: np.ndarray) -> List[int]:
    """
    Count how many boxes are needed to cover a trajectory for each box size.

    Parameters
    ----------
    path : np.ndarray
        The encoded weights to estimate the fractal dimension of.
    sizes : np.ndarray
        The range of box sizes to use for counting.

    Returns
    -------
    list
        A list containing the number of boxes needed to cover the trajectory for
        each box size.
    """
    counts = []
    for size in sizes:
        # Calculate the number of boxes along each axis
        max_min_diff = np.max(path, axis=0) - np.min(path, axis=0)
        num_boxes_along_axes = np.ceil(max_min_diff / size).astype(int)
        # Total number of boxes is the product of the numbers along each axis
        total_boxes = np.prod(num_boxes_along_axes)
        counts.append(total_boxes)
    return counts


def fractal_dimension(path: np.ndarray) -> float:
    """
    Estimate the fractal dimension of a trajectory of encoded weights.

    Uses box-counting to estimate the fractal dimension of the trajectory. This
    method involves covering the trajectory with "boxes" of a certain size and
    counting how many boxes are needed to fully cover the path. The fractal
    dimension is then estimated by observing how this number changes as the size
    of the boxes is varied.

    Parameters
    ----------
    path : np.ndarray
        The encoded weights to estimate the fractal dimension of.

    Returns
    -------
    float
        The estimated fractal dimension of the trajectory.

    Interpretation
    --------------
    A higher fractal dimension suggests a more complex trajectory with more
    intricate patterns, while a lower fractal dimension indicates a smoother
    trajectory with less complex patterns.

    Note
    ----
    For more accurate and meaningful fractal dimension estimates, especially in
    high-dimensional spaces, more sophisticated techniques and denser sampling
    of the trajectory would be required.
    """

    # Define a range of box sizes (decreasing sizes)
    box_sizes = np.geomspace(1.0, 0.1, num=10)

    # Count how many boxes are needed to cover the trajectory for each size
    box_counts = box_count(path, box_sizes)

    # Perform a log-log linear regression to estimate the fractal dimension
    log_box_sizes = np.log(box_sizes)
    log_box_counts = np.log(box_counts)

    # Reshape for sklearn
    X = log_box_sizes.reshape(-1, 1)
    y = log_box_counts.reshape(-1, 1)

    # Linear regression
    reg = LinearRegression().fit(X, y)
    fractal_dimension_estimate = -reg.coef_[0][0]

    return fractal_dimension_estimate
