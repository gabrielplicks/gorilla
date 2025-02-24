from typing import Dict



def parse_input(input_string: str) -> Dict[str, str]:

    """Parses an input string in the form of `key1=val1;key2=val2;...;keyn=valn` and returns a dictionary with the corresponding key-value pairs.



    Args:

        input_string: The input string to parse.



    Returns:

        A dictionary with the key-value pairs from the input string.

    """

    pairs = input_string.split(';')

    output = {}

    for pair in pairs:

        key, value = pair.strip().split('=')

        output[key] = value



    return output

