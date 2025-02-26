from typing import List



def string_to_upper_list(input_string: str) -> List[str]:

    """Converts a string to a list of uppercase words.



    Args:

        input_string: The input string to be converted.



    Returns:

        A list of uppercase words.

    """

    words = input_string.split()

    return [word.upper() for word in words]

