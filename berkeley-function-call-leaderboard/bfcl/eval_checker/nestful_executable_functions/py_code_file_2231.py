from typing import List, Union



def check_list_equality(list1: Union[List, None], list2: Union[List, None]) -> bool:

    """Checks if two lists are equal.



    Args:

        list1: The first list.

        list2: The second list.



    Returns:

        True if the two lists are equal, False otherwise.

    """

    if list1 is None or list2 is None:

        return False

    if list1 is not None and list2 is not None:

        if len(list1) == len(list2):

            for i in range(len(list1)):

                if list1[i] != list2[i]:

                    return False

            return True

    return False

