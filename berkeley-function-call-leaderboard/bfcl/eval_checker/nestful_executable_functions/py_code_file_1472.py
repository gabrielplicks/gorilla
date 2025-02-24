from typing import Dict, List



def filter_fields(data: Dict[str, any], keys: List[str]) -> Dict[str, any]:

    """Filters out all fields that are not in a set of keys.



    Args:

        data: The dictionary to filter.

        keys: The keys to keep in the filtered dictionary.



    Returns:

        A new dictionary with only the fields in `keys`.

    """

    return {key: data[key] for key in data if key in keys}

