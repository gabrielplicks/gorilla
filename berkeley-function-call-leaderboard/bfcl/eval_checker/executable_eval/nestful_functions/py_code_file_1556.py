from typing import Any, Dict, Union



def find_value_in_nested_dictionary(dict: Dict[str, Union[str, Dict[str, Any]]], value: str) -> Union[str, None]:

    """Finds the first key that contains `value` in its value in a nested dictionary.



    Args:

        dict: The dictionary to search.

        value: The value to search for.



    Returns:

        The first key that contains `value` in its value, or `None` if no key contains `value`.

    """

    return next((key for key, val in dict.items() if value in val), None)

