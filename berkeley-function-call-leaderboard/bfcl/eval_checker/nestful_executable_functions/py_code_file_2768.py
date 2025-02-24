from typing import Dict



def replace_nested_empty_dicts(d: Dict) -> Dict:

    """Replaces all nested empty dictionaries with the string "Empty" in the original dictionary.



    Args:

        d: The input dictionary.



    Returns:

        The modified dictionary.

    """

    if not d:

        return "Empty"



    for key, value in d.items():

        if isinstance(value, dict) and not value:

            d[key] = "Empty"

        elif isinstance(value, dict):

            d[key] = replace_nested_empty_dicts(value)



    return d

