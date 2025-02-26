from typing import AnyStr



def string_to_binary_ascii(string: AnyStr) -> str:

    """Converts a string to a binary representation of its ASCII values.

    Args:

        string: The input string.

    Returns:

        A string containing the binary representation of the ASCII values of the input string.

    """

    binary_ascii = ""

    for char in string:

        char_ascii = ord(char)

        char_binary = bin(char_ascii)[2:]

        binary_ascii += char_binary + " "

    return binary_ascii.strip()

