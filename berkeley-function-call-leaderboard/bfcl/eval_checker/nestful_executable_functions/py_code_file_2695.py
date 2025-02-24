from typing import List



def find_diff(list1: List[str], list2: List[str]) -> List[str]:

    """Finds the strings that exist in the first list but not in the second list.



    Args:

        list1: The first list of strings.

        list2: The second list of strings.



    Returns:

        A list of strings that exist in the first list but not in the second list.

    """

    return [s for s in list1 if s not in set(list2)]

