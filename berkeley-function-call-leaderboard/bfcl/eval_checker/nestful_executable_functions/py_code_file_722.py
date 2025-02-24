from typing import Dict, List



def extract_nested_data(input_string: str) -> Dict[str, List[str]]:

    """Extracts a nested data structure from a given string.



    Args:

        input_string: The input string to extract the data from.



    Returns:

        A dictionary containing the extracted data.

    """

    result = {}

    if len(input_string) == 0:  # Check if the string is empty

        return result

    elif len(input_string) == 1:  # Check if the string contains only a single character

        result[input_string] = []

        return result

    else:  # The string contains at least two characters

        result[input_string[0]] = []

        result[input_string[-1]] = []

        result[input_string[0]] = [char for char in input_string[1:-1]]

        return result

