import torch


def basic_variance(matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculate the variance of a matrix

    The variance of a weight matrix provides a measure of the spread of the
    values in the matrix. It is particularly relevant in understanding the
    stability of training and the propagation of signals through layers.

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix whose variance is to be calculated

    Returns
    -------
    torch.Tensor
        The variance of the matrix

    Interpretation
    --------------
    - Low Variance: Very small variance might suggest that the weights are too
        similar, which could lead to vanishing gradients during training.
    - High Variance: Conversely, very large variance might suggest that the
        weights are too different, potentially leading to exploding gradients.
    """
    return torch.var(matrix)
