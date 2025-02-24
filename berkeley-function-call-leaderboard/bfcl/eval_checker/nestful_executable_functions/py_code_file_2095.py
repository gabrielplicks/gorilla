from typing import List, Union



def are_all_same_type(obj_list: List[Union[str, int, float, List]]) -> bool:

    """Checks if all objects in the list are of the same type.



    Args:

        obj_list: A list of objects that can be either a string, integer, float, or a list.



    Returns:

        True if all objects in the list are of the same type, otherwise False.

    """

    if len(obj_list) == 0:

        return True

    prev_type = type(obj_list[0])

    for obj in obj_list[1:]:

        if type(obj) != prev_type:

            return False

    return True

