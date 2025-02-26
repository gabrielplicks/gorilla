import numpy as np

from typing import List, Tuple



def smooth_time_series(data: List[float], labels: List[str]) -> List[Tuple[float, str]]:

    """Computes the exponential moving average of the time series data with a smoothing factor of 0.5.

    Matches the index of each label in the label list with the index of the corresponding data point in the smoothed time series.

    Returns a list of tuples containing the data point and its corresponding label.



    Args:

        data: A list of time series data.

        labels: A list of time series data labels.

    """

    smoothed_data = np.convolve(data, [0.5, 0.5], mode='valid')

    smoothed_data_with_labels = [(smoothed_data[i], labels[i]) for i in range(len(smoothed_data))]

    return smoothed_data_with_labels

