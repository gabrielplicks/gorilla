from typing import Dict, Any



def replace_empty_values(dictionary: Dict[str, Any]) -> Dict[str, Any]:

    """Replaces empty/blank values in a dictionary with None.



    Args:

        dictionary: The dictionary to filter.



    Returns:

        A new dictionary with empty/blank values replaced with None.

    """

    def check_empty(value: Any) -> Any:

        if value in [None, "", " ", False]:

            return None

        else:

            return value



    filtered_dictionary = {key: check_empty(value) for key, value in dictionary.items()}



    return filtered_dictionary

