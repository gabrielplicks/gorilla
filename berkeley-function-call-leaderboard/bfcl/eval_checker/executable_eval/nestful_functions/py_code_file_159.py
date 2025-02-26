from typing import List, Dict, Any



def create_dictionary_from_lists(keys: List[str], values: List[Any]) -> Dict[str, Any]:

    """Creates a dictionary from two lists of keys and values.



    Args:

        keys: A list of keys to be used as dictionary keys.

        values: A list of values to be used as dictionary values.



    Returns:

        A dictionary where each key corresponds to a value.



    Raises:

        ValueError: If the lengths of the `keys` and `values` lists are not equal.

    """

    if len(keys) != len(values):

        raise ValueError("Lists must be of equal length")

    return dict(zip(keys, values))

