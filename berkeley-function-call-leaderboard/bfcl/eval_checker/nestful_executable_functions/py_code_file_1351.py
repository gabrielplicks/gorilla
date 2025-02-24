from typing import List



def positive_average(nums: List[float]) -> float:

    """Calculates the average of the positive numbers in a list.



    Args:

        nums: A list of numbers.



    Returns:

        The average of the positive numbers in the list. If the list is empty or contains only negative numbers, returns 0.

    """

    sum_positive = 0

    count_positive = 0



    for num in nums:

        if num > 0:

            sum_positive += num

            count_positive += 1



    if count_positive > 0:

        return sum_positive / count_positive

    else:

        return 0

