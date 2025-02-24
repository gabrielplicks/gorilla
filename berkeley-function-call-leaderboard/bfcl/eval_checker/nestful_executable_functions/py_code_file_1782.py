from typing import List



def str_to_binary(string: str) -> List[str]:

    """Converts a string to its binary representation, where each character is represented by the ASCII value in 8-bit binary form.

    Args:

        string: The input string to be converted to binary.

    Returns:

        A list of strings representing the binary representation of each character in the input string.

    """

    binary_list = [bin(ord(char))[2:].zfill(8) for char in string]

    return binary_list

