from typing import Dict



def parse_message(message: str) -> Dict[str, str]:

    """Parses a message string into a dictionary of key-value pairs.



    Args:

        message: A string containing key-value pairs separated by semicolons (;),

                 with each pair in the format `key=value`.



    Returns:

        A dictionary of the parsed message.

    """

    pairs = message.split(';')

    parsed_dict = {}

    for pair in pairs:

        key, value = pair.split('=')

        parsed_dict[key] = value

    return parsed_dict

