from typing import List



def split_without_empty_strings(input_string: str) -> List[str]:

    """Splits a string into different parts by whitespace and excludes those parts that contain only whitespace.

    Args:

        input_string: The input string to be split.

    Returns:

        A list of strings.

    """

    result = input_string.split()

    return [part.strip() for part in result if part.strip()]

