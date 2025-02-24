import numpy as np



def pad_duplicate_signals(signals: List[np.ndarray], padding_factor: int) -> List[np.ndarray]:

    """

    Pads each signal in a list of signals with a specified number of duplicates.



    Args:

        signals: A list of 2-dimensional numpy arrays.

        padding_factor: The number of times each signal is padded with its duplicate.



    Returns:

        A new list of signals, each padded with the specified number of duplicates.

    """

    padded_signals = []

    for signal in signals:

        for _ in range(padding_factor):

            signal = np.concatenate([signal, signal])

        padded_signals.append(signal)

    return padded_signals

