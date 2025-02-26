from typing import List



def moving_average(data: List[float], window_size: int) -> List[float]:

    """Calculates the moving average of a given time series data.

    Args:

        data: A list of numbers representing the values in the time series.

        window_size: The number of values to average.

    """

    return [

        sum(data[i : i + window_size]) / len(data[i : i + window_size])

        for i in range(len(data) - window_size + 1)

    ]

