from typing import List



def ratio_saliency(time_series: List[int]) -> List[float]:

    """Calculates the saliency ratio for a given time series.



    Args:

        time_series: A list of integers representing the time series.



    Returns:

        A list of floats representing the saliency ratios.

    """

    return [time_series[i] / sum(time_series) for i in range(len(time_series))]

