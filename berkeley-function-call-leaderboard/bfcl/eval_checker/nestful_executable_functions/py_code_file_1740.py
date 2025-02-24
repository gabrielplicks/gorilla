from typing import List



def remove_zeros(numbers: List[int]) -> None:

    """Removes all occurrences of zero from a list of integers.



    Args:

        numbers: A list of integers.



    Raises:

        IndexError: If an index out of range error occurs.

    """

    try:

        for i in range(len(numbers)):

            if numbers[i] == 0:

                del numbers[i]

    except IndexError:

        pass

