from typing import Dict



def parse_name_value_pairs(string: str) -> Dict[str, str]:

    """Parses a string containing name-value pairs with the format `name1=value1;name2=value2;...` and returns a dictionary of the parsed name-value pairs.



    Args:

        string: The input string containing name-value pairs.



    Returns:

        A dictionary of the parsed name-value pairs.

    """

    pairs = string.split(';')

    dictionary = {}

    for pair in pairs:

        if '=' in pair:

            name, value = pair.split('=')

            dictionary[name] = value

    return dictionary

