from typing import Union



def num_to_binary(n: Union[int, float]) -> Union[str, None]:

    """Converts a positive integer into its binary representation.

    Args:

        n: The input number.

    Returns:

        The binary representation of the input number as a string.

        If the input is not a positive integer, the function returns 'Invalid input'.

    """

    if n < 0:

        return 'Invalid input'

    else:

        return bin(n)[2:]

