import numpy as np

from typing import List



def remove_noise(signals: List[np.ndarray]) -> tuple:

    """Calculates the mean and variance of a set of noisy Poisson signals.

    Args:

        signals: A list of numpy arrays, where each array represents a noisy Poisson signal.

    Returns:

        A tuple containing the mean and variance of the subtracted signals.

    """

    means = [np.mean(signal) for signal in signals]

    variances = [np.var(signal) for signal in signals]

    subtracted_signals = [signal - mean for signal, mean in zip(signals, means)]

    subtracted_mean = np.mean(subtracted_signals)

    subtracted_var = np.var(subtracted_signals)

    return subtracted_mean, subtracted_var

