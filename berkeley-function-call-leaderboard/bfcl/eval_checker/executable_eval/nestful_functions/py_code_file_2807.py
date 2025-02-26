from typing import List, Dict



def sort_by(data: List[Dict[str, int]], key: str) -> List[Dict[str, int]]:

    """Sorts a list of dictionaries based on the value of a specified key.



    Args:

        data: A list of dictionaries where each dictionary has a key-value pair where the key is a string and the value is an integer.

        key: The key to sort the list on.

    """

    return sorted(data, key=lambda x: x[key])

