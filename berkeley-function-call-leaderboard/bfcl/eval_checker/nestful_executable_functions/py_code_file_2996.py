import re

from typing import Dict



def parse_strings_into_dict(text: str) -> Dict[str, str]:

    """Parses a string into a dictionary based on a specific format.



    Given a string `text` that contains colon-separated key-value pairs where the key contains only alphanumeric characters and the value is an alphanumeric string or a string literal, the function returns a dictionary where the keys are the first part of each pair and the values are the second part of each pair.



    Args:

        text: The input string to be parsed.



    Returns:

        A dictionary containing the key-value pairs extracted from the input string.

    """

    pairs = text.split(':')

    pattern = re.compile(r'^[a-zA-Z0-9]+$')

    dictionary = {}



    for pair in pairs:

        key, value = pair.split('=')

        if re.match(pattern, key):

            dictionary[key] = value



    return dictionary

