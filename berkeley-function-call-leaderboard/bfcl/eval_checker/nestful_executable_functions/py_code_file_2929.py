from typing import Union



def bitwise_reorder(s: Union[str, bytes]) -> int:

    """Calculates the integer representation of a binary string using bitwise reordering.

    Args:

        s: A string of 1s and 0s.

    """

    n = len(s)

    return sum(int(char) << n - i - 1 for i, char in enumerate(s))

