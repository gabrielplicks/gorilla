from typing import List, Dict



def dict_zip(list1: List[str], list2: List[str]) -> List[Dict[str, str]]:

    """Creates a list of dictionaries from two input lists.

    Each dictionary corresponds to an item in the two input lists,

    with the key of the dictionary being the value from the first list

    and the value being the value from the second list.

    Args:

        list1: The first input list.

        list2: The second input list.

    """

    return [{k: v} for k, v in zip(list1, list2)]

