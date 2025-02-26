import re

from typing import List



def parse_dicts(strings: List[str]) -> List[dict]:

    """Parses a list of strings that represent dictionaries and returns a list of dictionaries.



    Each dictionary in the output list has matching key-value pairs based on the input string format.



    Args:

        strings: A list of strings that represent dictionaries.



    Returns:

        A list of dictionaries.

    """

    dicts = []

    pattern = r"\s*'(\w+)':\s*(\d+)\s*(?:,|}$)"

    for string in strings:

        matches = re.findall(pattern, string)

        dictionary = {key: int(value) for key, value in matches}

        dicts.append(dictionary)

    return dicts

