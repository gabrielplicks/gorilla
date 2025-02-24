from typing import Union



def sum_of_all_continuous_sequences(n: Union[int, float]) -> int:

    """Calculates the sum of all continuous sequences from 1 to n.



    A continuous sequence is defined as a set of consecutive positive integers that add up to a specific value,

    where the sequence length can vary.



    Args:

        n: A positive integer representing the upper limit of the continuous sequences.



    Returns:

        The sum of all continuous sequences from 1 to n.

    """

    total_sum = (n * (n + 1)) // 2



    for i in range(1, n + 1):

        total_sum += (i * (i + 1)) // 2 - i



    return total_sum

