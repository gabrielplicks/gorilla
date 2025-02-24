from collections import OrderedDict

from typing import Dict, Any



def reverse_keys(input_dict: Dict[str, Any]) -> Dict[str, Any]:

    """Reverses the order of the top-level keys in a dictionary, and maintains the order of the nested dictionaries.



    Args:

        input_dict: The dictionary to reverse the order of the top-level keys.



    Returns:

        A dictionary with the top-level keys in reversed order.

    """

    return OrderedDict(sorted(input_dict.items(), reverse=True))

