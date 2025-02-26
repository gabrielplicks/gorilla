from typing import Union



def reverse_string_encoding(s: Union[str, bytes]) -> str:

    """Reverses the encoding of the input string.



    Args:

        s: The input string to reverse the encoding of.



    Raises:

        TypeError: If the input is not a string.

    """

    if not isinstance(s, str):

        raise TypeError("Input must be a string.")

    return s.encode().decode('utf-8', 'ignore')[::-1]

