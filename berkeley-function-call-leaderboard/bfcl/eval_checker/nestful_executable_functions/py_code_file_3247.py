from typing import List



def min_max_sum(integers_list: List[int]) -> int:

    """Calculates the sum of a list of integers, provided the minimum value is at least 0 and the maximum value is at most 255.

    If the minimum is less than 0 or the maximum value is greater than 255, returns -1.

    Args:

        integers_list: A list of integers.

    """

    min_value = min(integers_list)

    max_value = max(integers_list)



    if min_value < 0 or max_value > 255:

        return -1



    return sum(integers_list)

