from typing import List



def count_bins(values: List[float], counts: List[int], low: float, high: float) -> int:

    """Computes the number of bins that contain at least one non-zero count in a range.



    Args:

        values: The values of the bins.

        counts: The number of times that each value has been seen.

        low: The lower bound of the range.

        high: The upper bound of the range.

    """

    count_map = {}



    for i in range(len(values)):

        if values[i] >= low and values[i] <= high:

            if values[i] in count_map:

                count_map[values[i]] += counts[i]

            else:

                count_map[values[i]] = counts[i]



    return len(count_map)

