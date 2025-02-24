from typing import Union



def format_string_with_padding(string: str, padding_length: Union[int, float]) -> str:

    """Formats a string with padding.



    Args:

        string: The original string.

        padding_length: The total length of the string with padding.

    """

    return f"{string:>{padding_length}}"

