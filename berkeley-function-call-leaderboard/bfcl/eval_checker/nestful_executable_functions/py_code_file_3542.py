from typing import Union



def find_negative_index(sequence: Union[str, list, tuple], num: int) -> int:

    """Returns the negative index of a given number from the end of a sequence.

    Args:

        sequence: The sequence to search in.

        num: The number to find.

    """

    try:

        index = sequence.index(num)

        return -index - 1

    except ValueError:

        return -1

