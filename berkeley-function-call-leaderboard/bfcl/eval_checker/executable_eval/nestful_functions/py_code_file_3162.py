import numpy as np



def calculate_peak_amplitude(data_points: np.ndarray, sampling_rate: np.float64) -> np.float64:

    """Calculates the peak amplitude of a time series.



    Args:

        data_points: A NumPy array representing the time series data.

        sampling_rate: A float representing the sampling frequency.

    """

    data_points = np.asarray(data_points)

    time_diffs = np.diff(data_points, axis=0) / sampling_rate

    peak_amplitudes = np.abs(time_diffs)

    return peak_amplitudes.max()

