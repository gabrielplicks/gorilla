from typing import Dict



def parse_fields(input_str: str) -> Dict[str, str]:

    """Parses a string into a dictionary of key-value pairs.

    Args:

        input_str: A string containing key-value pairs separated by commas.

    """

    data = {}

    for pair in input_str.split(','):

        key, value = pair.split('=', 1)

        key = key.strip()

        value = value.strip()

        if not key or not value:

            continue  # Skip invalid key-value pairs

        data[key] = value

    return data

