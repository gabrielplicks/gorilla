from typing import List



def get_sequence_sum(n: int) -> List[int]:

    """Calculates the sum of the arithmetic sequence of N integers (starting from 1) and returns it as an array.



    Args:

        n: The number of integers in the sequence.



    Returns:

        A list of integers representing the sum of the arithmetic sequence for each integer from 1 to N.

    """

    sequence_sum = []

    for i in range(1, n + 1):

        sum_i = (i * (1 + i)) // 2

        sequence_sum.append(sum_i)

    return sequence_sum

