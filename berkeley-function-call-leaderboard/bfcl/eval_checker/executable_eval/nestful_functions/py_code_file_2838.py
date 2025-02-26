from typing import Dict, Any



def dict_length(dict_input: Dict[Any, Any]) -> int:

    """Calculates the number of key-value pairs in a dictionary.



    Args:

        dict_input: The dictionary to calculate the length of.



    Returns:

        The number of key-value pairs in the dictionary, or None if the input is not a dictionary.

    """

    if not isinstance(dict_input, dict):

        return None

    return len(dict_input)

