from typing import Union



def convert_to_unicode_if_available(input_string: Union[str, bytes]) -> str:

    """Converts a string to unicode if possible.



    Args:

        input_string: The input string to convert to unicode.



    Returns:

        The input string as a unicode string.



    Raises:

        Exception: If the input string cannot be decoded as unicode.

    """

    if isinstance(input_string, str):

        return input_string

    try:

        return input_string.decode('utf-8')

    except UnicodeDecodeError:

        raise Exception('Failed to decode the input string as unicode')

