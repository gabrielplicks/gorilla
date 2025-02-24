from typing import Union



def str_to_bytes_implicit(s: str) -> Union[bytes, int]:

    """Converts a string to its byte representation implicitly.



    Args:

        s: The string to convert.



    Returns:

        The byte representation of the string.

    """

    return s.encode('utf-8')

