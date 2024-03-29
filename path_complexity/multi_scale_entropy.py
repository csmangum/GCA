from typing import List, Tuple

import numpy as np


def sampen(L: np.ndarray, m: int, r: float) -> float:
    """
    Calculate the sample entropy of a time series.

    Parameters
    ----------
    L : np.ndarray
        The time series to calculate sample entropy for.
    m : int
        The embedding dimension.
    r : float
        The tolerance.

    Returns
    -------
    float
        The sample entropy of the time series.
    """
    N = len(L)
    B = 0.0
    A = 0.0
    L = np.array(L)  # Ensure L is a numpy array for vectorized operations

    for i in range(N - m):
        template_m = L[i : i + m]
        for j in range(i + 1, N - m):
            template_n = L[j : j + m]
            # Now that L is a numpy array, this subtraction will be element-wise
            if np.max(np.abs(template_m - template_n)) <= r:
                B += 1
                if np.abs(L[i + m] - L[j + m]) <= r:
                    A += 1
    return -np.log(A / B) if B != 0 else 0


def mse(path: np.ndarray, m: int, r: float, max_scale: int) -> List[float]:
    """
    Calculate the multi-scale entropy of a time series.

    Parameters
    ----------
    path : np.ndarray
        The time series to calculate multi-scale entropy for.
    m : int
        The embedding dimension.
    r : float
        The tolerance.
    max_scale : int
        The maximum scale to calculate entropy for.

    Returns
    -------
    List[float]
        The list of entropies at each scale.
    """
    entropies = []
    N = len(path)

    for tau in range(1, max_scale + 1):
        # Create coarse-grained time series at scale tau
        coarse_grained = [np.mean(path[i : i + tau]) for i in range(0, N, tau)]
        coarse_grained = np.array(coarse_grained)  # Convert to numpy array for sampen
        # Calculate SampEn for this scale
        e = sampen(coarse_grained, m, r)
        entropies.append(e)

    return entropies


def multi_scale_entropy(
    path: np.ndarray, m: int = 2, r: float = 0.2, max_scale: int = 20
) -> Tuple[List[float], int]:
    """
    Calculate the multi-scale entropy of a time series.

    Parameters
    ----------
    path : np.ndarray
        The time series to calculate multi-scale entropy for.
    m : int, optional
        The embedding dimension, by default 2.
    r : float, optional
        The tolerance, by default 0.2.
    max_scale : int, optional
        The maximum scale to calculate entropy for, by default 20.

    Returns
    -------
    Tuple[List[float], int]
        A tuple containing the list of entropies at each scale and the maximum scale used.

    Interpretation
    --------------
    Higher entropy values indicate more complexity or irregularity at a given scale.
    Lower entropy values indicate more regularity or predictability at a given scale.
    """

    entropies = mse(path, m, r, max_scale)

    return entropies, max_scale
