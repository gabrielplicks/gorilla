from typing import Union



def format_padding(string: str, padding_length: int, padding_char: Union[str, int]) -> str:

    """

    Builds a valid padding string of the required length by repeating the padding character.

    If the string is longer than the padding length, the function returns the original string without any padding.



    Args:

        string: The user-supplied string.

        padding_length: The required length of the padding string.

        padding_char: The padding character.

    """

    if not isinstance(string, str):

        return "Invalid string parameter"

    if not isinstance(padding_length, int):

        return "Invalid padding length"

    if not isinstance(padding_char, str) or len(padding_char) != 1:

        return "Invalid padding character"

    padding = padding_char * (padding_length - len(string))

    if len(string) > padding_length:

        return string

    return string + padding

