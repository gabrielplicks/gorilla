from typing import Union



def longest_consecutive_ones(s: Union[str, bytes]) -> int:

    """Calculates the length of the longest consecutive 1s sequence in a string of 1s and 0s.



    Args:

        s: A string of 1s and 0s.



    Returns:

        The length of the longest consecutive 1s sequence.

    """

    current = maximum = 0

    for bit in s:

        if bit == '1':

            current += 1

        else:

            current = 0

        maximum = max(maximum, current)

    return maximum

