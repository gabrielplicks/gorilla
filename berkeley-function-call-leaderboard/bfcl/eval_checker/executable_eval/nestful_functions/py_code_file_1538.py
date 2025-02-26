from typing import List



def two_sum_gte(values: List[int], threshold: int) -> bool:

    """Checks if there are any two values in the `values` collection that sum up to a value greater than or equal to `threshold`.



    Args:

        values: A sorted collection of positive integers.

        threshold: The threshold value.



    Returns:

        True if there are any two values in the `values` collection that sum up to a value greater than or equal to `threshold`. Otherwise, False.

    """

    left = 0

    right = len(values) - 1



    while left < right:

        sum_value = values[left] + values[right]

        if sum_value >= threshold:

            return True

        elif sum_value < threshold:

            left += 1

        else:

            right -= 1



    return False

