from typing import Dict



def dict_string(dictionary: Dict[str, int]) -> str:

    """Concatenates the keys and values of a dictionary into a single string.

    Args:

        dictionary: The dictionary to concatenate.

    """

    string = ""

    for key, value in dictionary.items():

        string += f"{key}: {value}, "

    if string:

        string = string[:-2]  # Remove trailing comma and space

    return string

