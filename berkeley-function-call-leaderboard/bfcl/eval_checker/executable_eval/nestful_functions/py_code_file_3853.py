from typing import Union



def encode_as_binary(x: Union[int, str]) -> str:

    """Encodes an integer or a string into a binary sequence.



    The binary sequence represents the integer's value in ASCII encoding.

    If the input is a string, it is first converted to an integer using the

    `ord()` function.



    Args:

        x: The integer or string to encode.



    Returns:

        The binary sequence representing the input.

    """

    if isinstance(x, str):

        x = ord(x)



    binary_string = bin(x)

    binary_sequence = binary_string[2:]

    if len(binary_sequence) % 8 != 0:

        binary_sequence = '0' * (8 - len(binary_sequence) % 8) + binary_sequence



    return binary_sequence

