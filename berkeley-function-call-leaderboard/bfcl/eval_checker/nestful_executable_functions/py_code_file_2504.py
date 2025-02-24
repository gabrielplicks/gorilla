from typing import Dict



def parse_string_dict(string_dict: str) -> Dict[str, str]:

    """Parses a dictionary of string-based key-value pairs.



    Args:

        string_dict: A string of key-value pairs, where the key and value are separated by a colon and a space.



    Returns:

        A dictionary with the keys and values.

    """

    pairs = string_dict.split(", ")

    dict = {}

    for pair in pairs:

        key, value = pair.split(": ")

        dict[key] = value

    return dict

