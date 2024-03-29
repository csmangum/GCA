import numpy as np


def path_efficiency(path: np.ndarray, loss_history: np.ndarray) -> float:
    """
    Calculate the path efficiency of a given path.

    Parameters
    ----------
    path : np.ndarray
        The path for which to calculate the path efficiency.
    loss_history : np.ndarray
        The history of loss values along the path during training.

    Returns
    -------
    float
        The path efficiency of the path.

    Interpretation
    --------------
    High Path Efficiency: Indicates that the model is effectively minimizing the
        loss with relatively small movements in the weight space. This could
        suggest a smooth optimization landscape or an effective training regimen.
    Low Path Efficiency: May indicate that the training process is taking a
        circuitous path through the weight space, possibly due to a complex
        optimization landscape, suboptimal learning rates, or other factors that
        could be adjusted for more efficient training.

    Note
    ----
    Path efficiency is a concept that evaluates the efficiency of a path taken
    by an algorithm or a model in reaching its goal. In the context of analyzing
    the trajectory of weight vectors in neural network training, path efficiency
    can measure how directly the training process moves towards minimizing the
    loss function relative to the path's length in the weight space. This metric
    can provide insights into the training process's effectiveness and the
    optimization landscape's complexity.
    """
    # Calculate the total path length
    path_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total_path_length = np.sum(path_lengths)

    # Calculate the overall loss decrease
    loss_decrease = loss_history[0] - loss_history[-1]

    # Calculate path efficiency
    path_efficiency = loss_decrease / total_path_length

    return path_efficiency
