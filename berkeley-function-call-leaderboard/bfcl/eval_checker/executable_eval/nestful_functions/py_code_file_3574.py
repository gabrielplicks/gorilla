from typing import Dict, List



def extract_over_100(dict: Dict[str, int]) -> List[str]:

    """Extracts keys from a dictionary that have a value greater than 100.



    Args:

        dict: The dictionary to extract keys from.



    Returns:

        A list of keys that have a value greater than 100.

    """

    return [key for key, value in dict.items() if value > 100]

