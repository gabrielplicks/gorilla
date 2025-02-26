from typing import List



def find_sub_list(list: List[int], sublist: List[int]) -> bool:

    """Checks if the sublist occurs in the list.



    Args:

        list: The list to search in.

        sublist: The sublist to search for.



    Returns:

        True if the sublist occurs in the list and False otherwise.

    """

    for i in range(len(list) - len(sublist) + 1):

        if list[i:i+len(sublist)] == sublist:

            return True

    return False

