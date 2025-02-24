from typing import Dict



def delete_empty_keys(d: Dict[str, str]) -> Dict[str, str]:

    """

    Deletes the keys with empty values from a dictionary.



    Args:

        d: The input dictionary.



    Returns:

        A new dictionary with all the keys that have an empty value removed.

    """

    return {k: v for k, v in d.items() if v != ''}

