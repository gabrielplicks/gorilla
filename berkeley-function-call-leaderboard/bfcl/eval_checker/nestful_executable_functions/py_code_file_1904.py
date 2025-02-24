from typing import List



def generate_power_of_two_list(n: int) -> List[int]:

    """Generates the first `n` terms of the "power of two" sequence.



    The sequence is defined as follows: the first term is 1, the second term is 2,

    the third term is 4, and so on.



    Args:

        n: A non-negative integer.



    Returns:

        A list of the first `n` terms of the "power of two" sequence.

    """

    power_of_two_list = []

    for i in range(n):

        power_of_two_list.append(2 ** i)

    return power_of_two_list

