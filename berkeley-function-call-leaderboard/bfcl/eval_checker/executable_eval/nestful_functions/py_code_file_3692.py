from typing import Dict, List, Union



def match_second_dict(dict1: Dict[int, Union[List[str], int]], dict2: Dict[int, int]) -> bool:

    """Checks if the second dictionary contains exactly one key with the same value as a list element in the first dictionary.



    Args:

        dict1: A dictionary mapping integers to lists of strings or integers.

        dict2: A dictionary mapping integers to integers.



    Returns:

        True if the second dictionary contains exactly one key with the same value as a list element in the first dictionary. Otherwise, False.

    """

    if len(dict2) == 1 and list(dict2.values())[0] in dict1.values():

        return True

    for values in dict1.values():

        if isinstance(values, list) and list(dict2.values())[0] in values:

            return True



    return False

