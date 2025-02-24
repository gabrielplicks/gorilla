from typing import Dict, Any



def remove_empty_values(dictionary: Dict[str, Any]) -> Dict[str, Any]:

    """Removes all key-value pairs from a dictionary where the value is an empty string or an empty list.



    Args:

        dictionary: The dictionary to remove empty values from.



    Returns:

        A new dictionary with all empty values removed.

    """

    return dict(filter(lambda item: item[1] not in ['', []], dictionary.items()))

