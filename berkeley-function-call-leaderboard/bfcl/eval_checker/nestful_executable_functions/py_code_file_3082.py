from typing import Tuple



def concatenate_without_plus(filepath: str, string_to_append: str) -> str:

    """Concatenates a string read from a file with another string without using the '+' operator.

    The file is opened in the utf-8 encoding.



    Args:

        filepath: The path to the file to read from.

        string_to_append: The string to append to the string read from the file.



    Returns:

        The concatenated string.

    """

    with open(filepath, encoding='utf-8') as f:

        string_to_prepend = f.read()



    return ''.join([string_to_prepend, string_to_append])

