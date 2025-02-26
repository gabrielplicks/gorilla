from typing import Dict, List, Any



def convert_to_list_of_dicts(d: Dict[str, List[int]]) -> List[Dict[str, Any]]:

    """Converts a dictionary into a list of dictionaries.

    Each entry in the list corresponds to a key-value pair in the input dictionary.

    Args:

        d: The input dictionary.

    Returns:

        A list of dictionaries.

    """

    return [{"key": key, "value": value} for key, value in d.items()]

