from typing import Dict, List



def extract_subset(d: Dict[str, any], keys: List[str]) -> Dict[str, any]:

    """Extracts a subset of a given dictionary, where the subset contains only the entries that match specific criteria.



    Args:

        d: The input dictionary.

        keys: The list of keys to be extracted.



    Returns:

        A new dictionary containing only the entries from `d` where the keys are present in `keys`.

    """

    return {key: value for key, value in d.items() if key in keys}

