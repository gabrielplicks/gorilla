from typing import List



def find_avg_even(array: List[int]) -> float:

    """Computes the average of all the even numbers from an input array of positive integers.



    Args:

        array: The input array of positive integers.



    Returns:

        The average of all the even numbers in the array.

    """

    total = 0

    count = 0

    for num in array:

        if num % 2 == 0:

            total += num

            count += 1

    return total / count if count > 0 else 0

