from typing import Dict, List, Tuple



def parse_lines(lines: List[str]) -> Dict[str, List[str]]:

    """Parses a list of lines containing key-value pairs and returns a dictionary.



    Each line in the input list is a string containing a key-value pair separated by a colon.

    The function returns a dictionary where the keys are the keys from the input lines,

    and the values are a list of all values associated with the given key.



    Args:

        lines: A list of strings, where each string contains a key-value pair separated by a colon.



    Returns:

        A dictionary where the keys are the keys from the input lines, and the values are a list

        of all values associated with the given key.

    """

    dictionary = {}

    for line in lines:

        key, value = line.split(':')

        dictionary.setdefault(key, []).append(value)

    return dictionary

