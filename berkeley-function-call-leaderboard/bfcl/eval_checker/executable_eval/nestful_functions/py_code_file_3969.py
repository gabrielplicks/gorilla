from typing import Dict, Any



def get_dict_values_as_string(d: Dict[str, Any]) -> str:

    """Returns a concatenated string representation of the values of a dictionary, separated by a comma and space.

    The string is surrounded by a pair of parentheses, and the keys of the dictionary are considered optional.



    Args:

        d: The input dictionary.



    Returns:

        A string with keys and values separated by commas and surrounded by parentheses.

    """

    return '(' + ', '.join(map(str, d.values())) + ')'

