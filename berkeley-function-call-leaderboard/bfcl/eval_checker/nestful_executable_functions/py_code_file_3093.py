from typing import List



def percentiles(data: List[int]) -> tuple:

    """Calculates the 10th and 90th percentiles of a list of integers.



    Args:

        data: A list of integers.



    Returns:

        A tuple containing the 10th and 90th percentiles.

    """

    data.sort()

    idx_10 = int(0.1 * len(data))

    idx_90 = int(0.9 * len(data))

    return data[idx_10], data[idx_90]

