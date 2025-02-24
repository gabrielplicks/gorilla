from typing import Dict, Any



def get_first_item(dictionary: Dict[Any, Any]) -> Tuple[Any, Any]:

    """

    Returns the first key-value pair found in the given dictionary.

    Args:

        dictionary: The input dictionary.

    Raises:

        AssertionError: If the dictionary is empty.

    """

    if len(dictionary) == 0:

        raise AssertionError("The dictionary is empty.")

    return next(iter(dictionary.items()))

