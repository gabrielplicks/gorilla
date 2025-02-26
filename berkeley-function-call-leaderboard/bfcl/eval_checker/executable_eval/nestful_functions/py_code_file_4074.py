from typing import Dict



def get_string_mapping(input_string: str) -> Dict[str, int]:

    """Creates a dictionary mapping each unique character in the input string to its index.



    Args:

        input_string: The input string.



    Returns:

        A dictionary mapping each unique character in the input string to its index.

    """

    mapping = {}

    for i, c in enumerate(input_string):

        if c not in mapping:

            mapping[c] = i

    return mapping

