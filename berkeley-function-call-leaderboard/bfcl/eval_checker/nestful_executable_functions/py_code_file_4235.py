from typing import List



def find_new_elements(old_list: List[int], new_list: List[int]) -> List[int]:

    """Identifies new elements that appear in new_list but not in old_list.



    Args:

        old_list: The original list of elements.

        new_list: The updated list of elements.



    Returns:

        A list of new elements that appear in new_list but not in old_list.

    """

    new_elements = set(new_list) - set(old_list)

    return list(new_elements)

