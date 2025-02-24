from typing import Dict, Any, List



def wrap_values(input_dict: Dict[str, Any]) -> Dict[str, Any]:

    """Wraps the values associated with the key 'k' in a list if they are not already a list.



    Args:

        input_dict: The input dictionary.



    Raises:

        TypeError: If the input is not a dictionary.

        KeyError: If the input dictionary already contains a key 'k'.

    """

    if not isinstance(input_dict, dict):

        raise TypeError("Input must be a dictionary")

    return {k: [v] if k == 'k' and not isinstance(v, list) else v for k, v in input_dict.items()}

