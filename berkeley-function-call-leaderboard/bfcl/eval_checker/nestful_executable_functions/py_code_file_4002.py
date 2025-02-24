from typing import Dict



def convert_string_to_json(string: str) -> Dict[str, str]:

    """Converts a string into a JSON-like object.



    Args:

        string: The input string to convert.



    Returns:

        A dictionary object representing the JSON-like object.



    Raises:

        ValueError: If the input string ends with a comma.

    """

    substrings = string.rstrip().split(", ")

    if substrings[-1].endswith(","):

        raise ValueError("Input string cannot end with a comma")

    pairs = [substring.split(": ") for substring in substrings]

    key_values = [(key, value.strip() if value.strip().isdigit() else f'"{value.strip()}"') for key, value in pairs]

    return dict(key_values)

