from typing import Tuple

import numpy as np


def spectral_analysis(path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform spectral analysis on a given path.

    Spectral analysis is a technique used to analyze the frequency content of a signal.

    Parameters
    ----------
    path : np.ndarray
        The path to perform spectral analysis on.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the power spectrum and the corresponding frequencies.

    Interpretation
    --------------
    High amplitude at a specific frequency indicates a strong periodic component
    at that frequency. By examining this spectrum, you can identify:

    Dominant frequencies: Indicating regular patterns or cycles in how the
        weights were updated.
    Noise: High-frequency components with low amplitude might indicate noise or
        irregularities in the training process.

    Note
    ----
    Spectral Analysis involves transforming data from the time domain into the
    frequency domain to identify the periodicities, trends, or unique components
    within the data. This method is particularly useful for analyzing the
    complexity or dynamics of a sequence, such as the trajectory of weight vectors
    in machine learning models, by revealing the frequency components present in
    the sequence.
    """

    # Perform the FFT
    fft_result = np.fft.fft(path)
    frequencies = np.fft.fftfreq(len(path))

    # Compute the power spectrum (squared magnitude of the FFT results)
    power_spectrum = np.abs(fft_result) ** 2

    return power_spectrum, frequencies
