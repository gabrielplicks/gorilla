from typing import List



def average_value(values: List[int]) -> int:

    """Calculates the average of a list of integers, rounded to the nearest integer.

    If the average value is a tie, returns the larger value of the two.



    Args:

        values: A list of integers.



    Returns:

        The average of the values, rounded to the nearest integer.

    """

    average = round(sum(values) / len(values))

    if average * 2 == sum(values):

        average += 1



    return average

