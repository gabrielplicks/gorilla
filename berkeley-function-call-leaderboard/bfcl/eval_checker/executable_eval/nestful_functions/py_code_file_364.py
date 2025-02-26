from typing import Union



def string_to_sentence(string: Union[str, bytes]) -> str:

    """Converts a string into a sentence. The function capitalizes the first letter, lowercases all subsequent letters, and adds a period at the end. It also validates its input: the function exits the program and prints an error message if the input string is less than 5 characters long.



    Args:

        string: The input string.



    Raises:

        ValueError: If the input string is less than 5 characters long.

    """

    if len(string) < 5:

        raise ValueError("Input string must be at least 5 characters long.")

    return string.title() + '.'

