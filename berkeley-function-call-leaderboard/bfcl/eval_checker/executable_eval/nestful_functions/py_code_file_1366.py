from typing import List



def calculate_midpoints(price_data: List[float], window_size: int) -> List[float]:

    """Calculates the midpoint values of asset price data over a specified window size.



    Args:

        price_data: A list of asset price data points.

        window_size: The size of the sliding window.



    Returns:

        A list of midpoint values.

    """

    midpoints = []

    for i in range(len(price_data) - window_size + 1):

        window_data = price_data[i:i+window_size]

        window_sum = sum(window_data)

        window_midpoint = window_sum / window_size

        midpoints.append(window_midpoint)



    return midpoints

