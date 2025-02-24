from typing import Union



def convert_bytes_string_to_string(input: Union[bytes, int]) -> str:

    """Converts a bytes string to a string representation.

    Args:

        input: The input bytes string or integer.

    """

    if isinstance(input, bytes):

        return input.decode('utf-8')

    else:

        return str(input)

