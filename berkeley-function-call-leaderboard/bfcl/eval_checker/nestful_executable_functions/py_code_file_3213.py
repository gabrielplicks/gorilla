from typing import List, Dict



def filter_by_key(data: List[Dict], key: str) -> List[Dict]:

    """Filters a list of dictionaries to return only those that contain a certain key.

    Args:

        data: A list of dictionaries.

        key: A string that represents a key name.

    """

    return [d for d in data if key in d]

