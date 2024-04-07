import torch


def basic_magnitude(matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Frobenius norm of a matrix

    The Frobenius norm of a weight matrix is a generalization of the Euclidean
    norm for matrices. It provides a measure of the overall magnitude of the
    matrix and is particularly relevant in understanding the stability of
    training and the propagation of signals through layers.

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix whose norm is to be calculated

    Returns
    -------
    torch.Tensor
        The Frobenius norm of the matrix

    Notes
    -----
    - Low Magnitudes: Very small values might suggest that the weights are too
        small, which could lead to vanishing gradients during training.
    - High Magnitudes: Conversely, very large values might suggest that the
        weights are too large, potentially leading to exploding gradients.
    """
    return torch.norm(matrix).item()


def spectral_norm(matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculate the spectral norm of a matrix

    The spectral norm of a weight matrix is the largest singular value of that
    matrix. It provides a measure of the maximum stretching factor of the matrix
    and is particularly relevant in understanding the stability of training and
    the propagation of signals through layers.

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix whose spectral norm is to be calculated

    Returns
    -------
    torch.Tensor
        The spectral norm of the matrix

    Interpretation
    --------------
    - High Spectral Norm: A high spectral norm indicates that the layer can
        significantly amplify the inputs or gradients. This might lead to exploding
        gradients during training, where small changes in inputs lead to
        disproportionately large changes in the output.
    - Low Spectral Norm: A low spectral norm suggests that the layer might suppress
        information, potentially leading to vanishing gradients, where changes in
        input have minimal impact on the output.
    - Balanced Norm: Ideally, you want the spectral norms to be balanced across
        layers. This balance helps in maintaining stable gradient flow through
        the network, preventing both vanishing and exploding gradients.
    """
    U, S, V = torch.svd(matrix)

    return S.max().item()


def condition_number(matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculate the condition number of a matrix

    The condition number of a matrix, often calculated as the ratio of the largest
    singular value to the smallest, measures how sensitive a function is to changes
    or errors in input. For weight matrices, a high condition number can indicate
    potential numerical stability issues, affecting gradient propagation.

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix whose condition number is to be calculated

    Returns
    -------
    torch.Tensor
        The condition number of the matrix

    Interpretation
    --------------
    - High Condition Number: Indicates that the matrix (and thus the layer) is
        ill-conditioned. This suggests that the network might be unstable with
        respect to input variations or during training, as it amplifies certain
        directions in the input space much more than others. This can make training
        difficult and lead to poor generalization.
    - Low Condition Number: Indicates that the matrix is well-conditioned, meaning
        the network should be relatively stable with respect to input variations
        and during training. It suggests that the layer treats all directions in
        the input space more uniformly, which is desirable for stable gradients
        and effective learning.
    - Very Low Condition Number: While generally good, if condition numbers are
        extremely low (close to 1), it might suggest that the layer is not
        effectively learning distinctions in the input data. However, this
        scenario is less common in practice compared to the issue of high
        condition numbers.
    """
    U, S, V = torch.svd(matrix)

    return (S.max() / S.min()).item()


def eigenvalue_calculation(matrix: torch.Tensor) -> torch.Tensor:
    """
    Calculate the eigenvalues of a matrix

    Analyzing the eigenvalues of a weight matrix (or more generally, the eigenvalues
    of the layer's transfer function) can provide insights into the dynamics of
    learning, including aspects like learning speed and stability. Layers with
    eigenvalues of their weight matrices close to zero can lead to vanishing
    gradients, while large eigenvalues might cause exploding gradients.

    Parameters
    ----------
    matrix : torch.Tensor
        The matrix whose eigenvalues are to be calculated

    Returns
    -------
    torch.Tensor
        The eigenvalues of the matrix

    Interpretation
    --------------
    The distribution of eigenvalues can help diagnose potential training issues.
    Ideally, eigenvalues should not be too small or too large, as this can affect
    the network's ability to learn efficiently.

    """
    return torch.linalg.eig(matrix).eigenvalues
