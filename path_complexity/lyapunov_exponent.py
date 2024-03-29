import numpy as np


def lyapunov_exponent(path: np.ndarray, epsilon: float = 1e-5) -> float:
    """
    Calculate the Lyapunov Exponent of a given path.

    Parameters
    ----------
    path : np.ndarray
        The path for which to calculate the Lyapunov Exponent.
    epsilon : float, optional
        The perturbation factor to apply to the path, by default 1e-5.

    Returns
    -------
    float
        The Lyapunov Exponent of the path.

    Interpretation
    --------------
    Positive Lyapunov Exponent: Indicates chaotic behavior, with trajectories
        diverging rapidly from each other. In training dynamics, this might
        suggest sensitivity to initial conditions or instability in learning.
    Negative Lyapunov Exponent: Suggests that trajectories converge, indicating
        stability and robustness to initial conditions.
    Zero Lyapunov Exponent: Indicates neutral behavior, where trajectories
        neither converge nor diverge significantly.

    Note
    ----
    The Lyapunov exponent is a measure used to quantify the rate at which
    trajectories in a dynamical system diverge or converge. In the context of
    machine learning models, particularly when analyzing the trajectory of weight
    vectors during training, the Lyapunov exponent can offer insights into the
    stability and sensitivity of the learning process to initial conditions.
    """
    perturbed_vectors = path + epsilon
    distances = np.linalg.norm(path - perturbed_vectors, axis=1)

    timesteps = np.arange(1, len(path))  # Start from 1 to match distances[1:]

    # Ensure distances are above a threshold to avoid log(0)
    threshold = 1e-10  # Small positive number to avoid log(0)
    valid_distances = np.maximum(distances[1:], threshold)  # Apply threshold

    log_distances = np.log(valid_distances)
    lyapunov_exponent, _ = np.polyfit(timesteps, log_distances, 1)

    return lyapunov_exponent
